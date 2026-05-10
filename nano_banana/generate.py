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


api_key = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

prompt = "Connect each numbered dot in the image with thin lines in numerical order. You should only change the image by overlaying lines on top of the image."

image_dirs = ["/Users/log/Github/sketchvlm/datasets/connect_dots_dataset/random_source", "/Users/log/Github/sketchvlm/datasets/connect_dots_dataset/outlines_source", "/Users/log/Github/sketchvlm/datasets/connect_dots_dataset/worksheets_source"]

results_file = "/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/results.json"

# Load existing results if file exists
# if os.path.exists(results_file):
#     with open(results_file, "r") as f:
#         results = json.load(f)
# else:
#     results = []

# for image_dir in image_dirs:
#     for image_path in os.listdir(image_dir):
#         if image_path.endswith(".png"):
#             image = Image.open(os.path.join(image_dir, image_path))
#             original_filename = os.path.splitext(os.path.basename(image_path))[0]

#             response = client.models.generate_content(
#                 model="gemini-2.5-flash-image-preview",
#                 contents=[prompt, image],
#             )

#             for part in response.candidates[0].content.parts:
#                 if part.text is not None:
#                     print(part.text)
#                 elif part.inline_data is not None:
#                     output_image = Image.open(BytesIO(part.inline_data.data))
#                     output_filename = f"/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/{original_filename}_nano_generated.png"
#                     output_image.save(output_filename)
#                     # print(f"Saved as: {output_filename}")

#             results.append({
#                 "original_image_path": os.path.join(image_dir, image_path),
#                 "original_filename": original_filename,
#                 "output_filename": output_filename,
#                 "timestamp": time.time(),
#                 "prompt": prompt,
#             })

#             with open(results_file, "w") as f:
#                 json.dump(results, f, indent=2)

# Load existing results if file exists
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = []


# Create a set of already processed image paths for quick lookup
processed_paths = {result["original_image_path"] for result in results}
# processed_paths = {}

# Collect all images to process
all_images = []
for image_dir in image_dirs:
    for image_path in os.listdir(image_dir):
        if image_path.endswith(".png") or image_path.endswith(".jpg"):
            full_path = os.path.join(image_dir, image_path)
            all_images.append(full_path)

# Filter out already processed images
images_to_process = [img for img in all_images if img not in processed_paths]

print(f"Total images: {len(all_images)}")
print(f"Already processed: {len(processed_paths)}")
print(f"Remaining to process: {len(images_to_process)}")

# Process only the remaining images
for full_image_path in images_to_process:
    image = Image.open(full_image_path)
    # if "boat" in full_image_path: # boat.png breaks it for some reason
    #     continue
    original_filename = os.path.splitext(os.path.basename(full_image_path))[0]
    output_filename = f"/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/{original_filename}_nano_generated.png"

    print(f"\nProcessing: {original_filename}")

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt, image],
    )
    print(response)
    image_saved = False
    
    try:
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                output_image = Image.open(BytesIO(part.inline_data.data))
                output_image.save(output_filename)
                print(f"Saved: {original_filename}_nano_generated.png")
                image_saved = True
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    if image_saved:
        results.append({
            "original_image_path": full_image_path,
            "original_filename": original_filename,
            "output_filename": output_filename,
            "timestamp": time.time(),
            "prompt": prompt,
        })

        # Save to JSON after each generation
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(f"WARNING: No image generated for {original_filename}")

# Generate HTML comparison file
print("\nGenerating HTML comparison file...")
html_output = "/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/comparison.html"
labels_output = "/Users/log/Github/sketchvlm/nano_generated/connect_dots_nano/labels.json"

# Load existing labels if they exist
labels = {}
if os.path.exists(labels_output):
    with open(labels_output, "r") as f:
        labels = json.load(f)

html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Connect Dots Comparison - Original vs Generated</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .stats {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            text-align: center;
        }
        .stats span {
            margin: 0 15px;
            font-size: 16px;
        }
        .comparison-row {
            display: flex;
            align-items: center;
            margin: 20px 0;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison-row.no-change { border-left: 5px solid #2196F3; }
        .comparison-row.hallucinated-points { border-left: 5px solid #FF9800; }
        .comparison-row.improper-lines { border-left: 5px solid #F44336; }
        .comparison-row.good { border-left: 5px solid #4CAF50; }
        .image-container {
            flex: 1;
            text-align: center;
            padding: 10px;
        }
        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-container h3 {
            margin: 10px 0 5px 0;
            color: #555;
            font-size: 14px;
        }
        .filename {
            color: #888;
            font-size: 12px;
            margin-top: 5px;
        }
        .arrow {
            font-size: 48px;
            color: #4CAF50;
            padding: 0 20px;
        }
        .labels {
            margin-left: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .label-btn {
            padding: 10px 20px;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            background-color: white;
            font-size: 14px;
            transition: all 0.3s;
        }
        .label-btn:hover {
            transform: scale(1.05);
        }
        .label-btn.active {
            font-weight: bold;
        }
        .label-btn.good { border-color: #4CAF50; color: #4CAF50; }
        .label-btn.good.active { background-color: #4CAF50; color: white; }
        .label-btn.no-change { border-color: #2196F3; color: #2196F3; }
        .label-btn.no-change.active { background-color: #2196F3; color: white; }
        .label-btn.hallucinated-points { border-color: #FF9800; color: #FF9800; }
        .label-btn.hallucinated-points.active { background-color: #FF9800; color: white; }
        .label-btn.improper-lines { border-color: #F44336; color: #F44336; }
        .label-btn.improper-lines.active { background-color: #F44336; color: white; }
    </style>
</head>
<body>
    <h1>Connect Dots Comparison - Original vs Generated</h1>
    <div class="stats" id="stats">
        <span>Total: <strong id="total">0</strong></span>
        <span>Good: <strong id="good-count">0</strong></span>
        <span>No Change: <strong id="no-change-count">0</strong></span>
        <span>Hallucinated Points: <strong id="hallucinated-points-count">0</strong></span>
        <span>Improper Lines: <strong id="improper-lines-count">0</strong></span>
        <span>Unlabeled: <strong id="unlabeled-count">0</strong></span>
    </div>
"""

row_index = 0
for result in results:
    original_path = result["original_image_path"]
    output_path = result["output_filename"]
    filename = result["original_filename"]

    # Get existing labels if any
    current_labels = labels.get(filename, [])
    if isinstance(current_labels, str):
        current_labels = [current_labels]
    label_class = " ".join(current_labels) if current_labels else ""

    # Check if output file exists
    if os.path.exists(output_path):
        html_content += f"""
    <div class="comparison-row {label_class}" id="row-{row_index}" data-filename="{filename}">
        <div class="image-container">
            <h3>Original</h3>
            <img src="file://{original_path}" alt="Original {filename}">
            <div class="filename">{filename}</div>
        </div>
        <div class="arrow">→</div>
        <div class="image-container">
            <h3>Generated</h3>
            <img src="file://{output_path}" alt="Generated {filename}">
            <div class="filename">{os.path.basename(output_path)}</div>
        </div>
        <div class="labels">
            <button class="label-btn good {'active' if 'good' in current_labels else ''}" onclick="setLabel('{filename}', 'good', {row_index})">✓ Good</button>
            <button class="label-btn no-change {'active' if 'no-change' in current_labels else ''}" onclick="setLabel('{filename}', 'no-change', {row_index})">No Change</button>
            <button class="label-btn hallucinated-points {'active' if 'hallucinated-points' in current_labels else ''}" onclick="setLabel('{filename}', 'hallucinated-points', {row_index})">Hallucinated Points</button>
            <button class="label-btn improper-lines {'active' if 'improper-lines' in current_labels else ''}" onclick="setLabel('{filename}', 'improper-lines', {row_index})">Improper Lines</button>
        </div>
    </div>
"""
        row_index += 1

html_content += f"""
    <script>
        let labels = {json.dumps(labels)};

        function setLabel(filename, label, rowIndex) {{
            // Initialize labels array for this filename if needed
            if (!labels[filename]) {{
                labels[filename] = [];
            }} else if (typeof labels[filename] === 'string') {{
                // Convert old string format to array
                labels[filename] = [labels[filename]];
            }}

            // Toggle label in array
            const index = labels[filename].indexOf(label);
            if (index > -1) {{
                labels[filename].splice(index, 1);
            }} else {{
                labels[filename].push(label);
            }}

            // Remove filename if no labels
            if (labels[filename].length === 0) {{
                delete labels[filename];
            }}

            // Update UI
            const row = document.getElementById('row-' + rowIndex);
            const currentLabels = labels[filename] || [];
            row.className = 'comparison-row ' + currentLabels.join(' ');

            // Update button states
            const buttons = row.querySelectorAll('.label-btn');
            buttons.forEach(btn => {{
                const btnLabel = btn.classList[1]; // Get the label class (good, no-change, etc)
                if (currentLabels.includes(btnLabel)) {{
                    btn.classList.add('active');
                }} else {{
                    btn.classList.remove('active');
                }}
            }});

            // Auto-save to localStorage
            autoSaveLabels();

            // Update stats
            updateStats();
        }}

        function updateStats() {{
            const total = {len(results)};
            let good = 0, noChange = 0, hallucinated = 0, improper = 0;

            Object.values(labels).forEach(labelArray => {{
                const arr = Array.isArray(labelArray) ? labelArray : [labelArray];
                if (arr.includes('good')) good++;
                if (arr.includes('no-change')) noChange++;
                if (arr.includes('hallucinated-points')) hallucinated++;
                if (arr.includes('improper-lines')) improper++;
            }});

            const unlabeled = total - Object.keys(labels).length;

            document.getElementById('total').textContent = total;
            document.getElementById('good-count').textContent = good;
            document.getElementById('no-change-count').textContent = noChange;
            document.getElementById('hallucinated-points-count').textContent = hallucinated;
            document.getElementById('improper-lines-count').textContent = improper;
            document.getElementById('unlabeled-count').textContent = unlabeled;
        }}

        // Auto-save labels to localStorage
        function autoSaveLabels() {{
            localStorage.setItem('connect-dots-labels', JSON.stringify(labels));
        }}

        // Load labels from file on page load
        async function loadLabelsFromFile() {{
            try {{
                const response = await fetch('labels.json?t=' + Date.now());
                if (response.ok) {{
                    const fileLabels = await response.json();
                    labels = fileLabels;

                    // Update all rows to reflect loaded labels
                    Object.keys(labels).forEach((filename, idx) => {{
                        const labelArray = Array.isArray(labels[filename]) ? labels[filename] : [labels[filename]];
                        const rows = document.querySelectorAll('.comparison-row');
                        rows.forEach((row, rowIdx) => {{
                            if (row.dataset.filename === filename) {{
                                row.className = 'comparison-row ' + labelArray.join(' ');
                                const buttons = row.querySelectorAll('.label-btn');
                                buttons.forEach(btn => {{
                                    const btnLabel = btn.classList[1];
                                    if (labelArray.includes(btnLabel)) {{
                                        btn.classList.add('active');
                                    }}
                                }});
                            }}
                        }});
                    }});
                    updateStats();
                }}
            }} catch (err) {{
                // Fallback: load from localStorage
                const savedLabels = localStorage.getItem('connect-dots-labels');
                if (savedLabels) {{
                    labels = JSON.parse(savedLabels);
                    updateStats();
                }}
            }}
        }}

        // Save using File System Access API
        let fileHandle = null;

        async function saveLabelsToFile() {{
            try {{
                if (!fileHandle) {{
                    fileHandle = await window.showSaveFilePicker({{
                        suggestedName: 'labels.json',
                        types: [{{
                            description: 'JSON Files',
                            accept: {{'application/json': ['.json']}}
                        }}]
                    }});
                }}

                const writable = await fileHandle.createWritable();
                await writable.write(JSON.stringify(labels, null, 2));
                await writable.close();

                alert('Labels saved successfully!');
            }} catch (err) {{
                if (err.name !== 'AbortError') {{
                    console.error('Save failed:', err);
                    // Fallback to download
                    downloadLabels();
                }}
            }}
        }}

        // Fallback download function
        function downloadLabels() {{
            const dataStr = JSON.stringify(labels, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
            const linkElement = document.createElement('a');
            linkElement.setAttribute('href', dataUri);
            linkElement.setAttribute('download', 'labels.json');
            linkElement.click();
            alert('Labels downloaded. Please move the file to:\\n{html_output.rsplit("/", 1)[0]}/');
        }}

        // Load from file using File System Access API
        async function loadLabelsFromFilePicker() {{
            try {{
                [fileHandle] = await window.showOpenFilePicker({{
                    types: [{{
                        description: 'JSON Files',
                        accept: {{'application/json': ['.json']}}
                    }}],
                    multiple: false
                }});

                const file = await fileHandle.getFile();
                const contents = await file.text();
                labels = JSON.parse(contents);

                // Update UI
                location.reload();
            }} catch (err) {{
                if (err.name !== 'AbortError') {{
                    console.error('Load failed:', err);
                }}
            }}
        }}

        // Add buttons
        window.onload = function() {{
            loadLabelsFromFile();

            const statsDiv = document.getElementById('stats');

            const saveBtn = document.createElement('button');
            saveBtn.textContent = 'Save Labels';
            saveBtn.style.cssText = 'padding: 10px 20px; margin-left: 20px; cursor: pointer; background-color: #2196F3; color: white; border: none; border-radius: 4px;';
            saveBtn.onclick = saveLabelsToFile;
            statsDiv.appendChild(saveBtn);

            const loadBtn = document.createElement('button');
            loadBtn.textContent = 'Load Labels';
            loadBtn.style.cssText = 'padding: 10px 20px; margin-left: 10px; cursor: pointer; background-color: #FF9800; color: white; border: none; border-radius: 4px;';
            loadBtn.onclick = loadLabelsFromFilePicker;
            statsDiv.appendChild(loadBtn);

            updateStats();
        }};
    </script>
</body>
</html>
"""

with open(html_output, "w") as f:
    f.write(html_content)

print(f"HTML comparison saved to: {html_output}")
print(f"Total comparisons: {len(results)}")
