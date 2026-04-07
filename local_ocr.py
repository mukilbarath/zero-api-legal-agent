import warnings
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from PIL import Image

# ==========================================
# --- THE MONKEY PATCH FIXES ---
# ==========================================
# 1. Bypass the missing Token ID bug in modern Transformers
PretrainedConfig.forced_bos_token_id = None

# 2. Bypass the SDPA Attention memory bug
PreTrainedModel._supports_sdpa = False

# 3. Bypass the PyTorch Meta Tensor initialization bug
original_linspace = torch.linspace

def patched_linspace(*args, **kwargs):
    kwargs['device'] = 'cpu'
    return original_linspace(*args, **kwargs)

torch.linspace = patched_linspace
# ==========================================

# Suppress warnings for a clean terminal output
warnings.filterwarnings("ignore")

def extract_document_text(image_path):
    print("Loading Florence-2 model (Zero-API, 100% Local)...")
    
    model_id = "microsoft/Florence-2-base" 
    
    # Load model with specific flags to prevent backend crashes
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    print(f"Opening image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'. Please check the file path.")
        return None

    print("Scanning document layout and extracting text...")
    
    # Florence-2 specific task prompt for OCR
    task_prompt = "<OCR>"
    
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    
    # Generate text tokens
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,  # Token limit per page
        early_stopping=False,
        do_sample=False,
        num_beams=3           
    )
    
    # Decode tokens to raw text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Clean up the output using the processor's built-in parser
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    extracted_text = parsed_answer['<OCR>']
    
    print("\n--- Extracted Text ---")
    print(extracted_text)
    
    return extracted_text

if __name__ == "__main__":
    # Replace with your actual image file name
    extract_document_text("handwritten-text-1.jpg")