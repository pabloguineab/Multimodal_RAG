import streamlit as st
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores import utils as chromautils
from langchain.embeddings import HuggingFaceEmbeddings
import torchvision.transforms as transforms

# Configuración de claves API
os.environ["UNSTRUCTURED_API_KEY"] = "nCWjslXLCFnyDjbVqnOS9LAWbeWZ10"
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

# Función para procesar documentos
def process_document(uploaded_file):
    client = UnstructuredClient(api_key_auth=os.environ.get("UNSTRUCTURED_API_KEY"))
    files = shared.Files(
        content=uploaded_file.read(),
        file_name=uploaded_file.name,
    )
    req = shared.PartitionParameters(
        files=files,
        chunking_strategy="by_title",
        max_characters=512,
    )
    try:
        resp = client.general.partition(req)
    except SDKError as e:
        st.error(f"Error processing document: {e}")
        return None
    elements = dict_to_elements(resp.elements)
    return elements

# Función para formatear documentos
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Función para hacer inferencia
def do_inference_with_retriever(model, processor, retriever, prompts, max_new_tokens=50):
    retriever_results = retriever(prompts[0])
    context = format_docs(retriever_results)
    complete_prompt = context + "\n\n" + prompts[1]
    
    tokenizer = processor.tokenizer
    bad_words_ids = []  # Inicializar bad_words_ids
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor([complete_prompt], return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        eos_token_id=[eos_token_id],
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        early_stopping=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Interfaz de Streamlit
st.title("Document Processing and Chat Interface")

# Inicializar variables globales
model = None
processor = None
retriever = None

# Cargar documento
uploaded_file = st.file_uploader("Choose a document", type=["pdf", "pptx"])
if uploaded_file is not None:
    elements = process_document(uploaded_file)
    if elements is not None:
        st.write("Document processed successfully!")
        documents = []
        for element in elements:
            metadata = element.metadata.to_dict()
            page_content = getattr(element, "text", "")  # Acceder de forma segura a "text"
            documents.append(Document(page_content=page_content, metadata=metadata))
        docs = chromautils.filter_complex_metadata(documents)
        
        # Asegúrate de inicializar ChromaDB correctamente
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        try:
            db = Chroma.from_documents(docs, embeddings)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            st.write("Documents added to ChromaDB!")
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {e}")
    else:
        st.error("Failed to process document.")

# Entrenamiento y Fine-Tuning
if st.button("Train Model"):
    checkpoint = "HuggingFaceM4/idefics-9b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["lm_head", "embed_tokens"]
    )
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto")
    
    ds = load_dataset("TheFusion21/PokemonCards")
    ds = ds["train"].train_test_split(test_size=0.002)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    def convert_to_rgb(image):
        if image.mode == "RGB":
            return image
        image_rgba = image.convert("RGBA")
        background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, image_rgba)
        alpha_composite = alpha_composite.convert("RGB")
        return alpha_composite

    def ds_transforms(example_batch):
        image_size = processor.image_processor.image_size
        image_mean = processor.image_processor.image_mean
        image_std = processor.image_processor.image_std
        image_transform = transforms.Compose([
            convert_to_rgb,
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std)
        ])
        prompts = []
        for i in range(len(example_batch['caption'])):
            caption = example_batch['caption'][i].split(".")[0]
            prompts.append(caption)
        inputs = processor(prompts, images=[image_transform(image) for image in example_batch['image']], return_tensors="pt").to(device)
        inputs["labels"] = inputs["input_ids"]
        return inputs

    train_ds.set_transform(ds_transforms)
    eval_ds.set_transform(ds_transforms)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    training_args = TrainingArguments(
        output_dir=f"{checkpoint}-PokemonCards",
        learning_rate=2e-4,
        fp16=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        dataloader_pin_memory=False,
        save_total_limit=3,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=25,
        max_steps=25,
        logging_steps=5,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        load_best_model_at_end=False,
        report_to="none",
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds
    )
    trainer.train()
    st.write("Model trained successfully!")

# Realizar una pregunta
question = st.text_input("Ask a question")
image_url = st.text_input("Image URL (if any)")
if st.button("Submit"):
    if model is not None and processor is not None and retriever is not None:
        prompts = [
            image_url,
            f"Question: {question} Answer:"
        ]
        answer = do_inference_with_retriever(model, processor, retriever, prompts, max_new_tokens=100)
        st.write(f"Answer: {answer}")
    else:
        st.error("Model, processor, or retriever not initialized. Please process a document and train the model first.")
