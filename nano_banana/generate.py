from google import genai
from IPython.display import display, Markdown, Image
from google import genai
import os
import PIL.Image
from io import BytesIO
from IPython.display import display
from google.genai import types
from PIL import Image
from io import BytesIO
import time
import json


client = genai.Client()

prompt = "Connect each numbered dot in the image with thin lines in numerical order. You should only change the image by overlaying lines on top of the image."

image_dirs = ["/Users/log/Github/sketchvlm/datasets/connect_dots_dataset/random_source", "/Users/log/Github/sketchvlm/datasets/connect_dots_dataset/outlines_source", "/Users/log/Github/sketchvlm/datasets/connect_dots_dataset/worksheets_source"]

results = []
for image_dir in image_dirs:
    for image_path in os.listdir(image_dir):
        if image_path.endswith(".png"):
            image = Image.open(os.path.join(image_dir, image_path))
            original_filename = os.path.splitext(os.path.basename(image_path))[0]

            response = client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[prompt, image],
            )

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
                elif part.inline_data is not None:
                    output_image = Image.open(BytesIO(part.inline_data.data))
                    output_filename = f"/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/{original_filename}_nano_generated.png"
                    output_image.save(output_filename)
                    # print(f"Saved as: {output_filename}")

            results.append({
                "original_filename": original_filename,
                "output_filename": output_filename,
                "timestamp": time.time(),
                "prompt": prompt,
            })


with open("/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/results.json", "w") as f:
    json.dump(results, f)