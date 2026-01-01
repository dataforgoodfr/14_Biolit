import csv
import json
import argparse
import urllib.request
import os
import time
import subprocess
from pathlib import Path

def convert_csv_to_labelstudio(csv_file, output_file, limit=None, force=False):
    tasks = []

    # Create images directory
    images_dir = Path('images')
    images_dir.mkdir(exist_ok=True)

    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')

        for row in reader:
            # Check limit
            if limit and len(tasks) >= limit:
                break

            # Get first image URL
            images = row.get('images - observation', '')
            if not images:
                continue

            # Split by pipe and take first URL
            image_url = images.split(' | ')[0].strip()

            # Get common name
            nom_commun = row.get('Nom commun - observation', '').strip()

            # Skip if no name
            if not nom_commun:
                continue

            # Download image
            filename = Path(image_url).name
            local_path = images_dir / filename

            # Skip if file exists and not forcing
            if local_path.exists() and not force:
                print(f"Skipping {filename} (already exists)")
            else:
                try:
                    print(f"Downloading {image_url}...")
                    urllib.request.urlretrieve(image_url, local_path)
                    time.sleep(1)  # Be nice to the server
                except Exception as e:
                    print(f"Failed to download {image_url}: {e}")
                    continue

            # Create task with annotation using localhost URL
            task = {
                "data": {
                    "captioning": f"http://localhost:8000/images/{filename}"
                },
                "annotations": [{
                    "result": [
                        {
                            "value": {
                                "choices": [nom_commun]
                            },
                            "from_name": "choice",
                            "to_name": "image",
                            "type": "choices"
                        }
                    ]
                }]
            }

            tasks.append(task)

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(tasks)} tasks to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download images, format CSV to Label Studio JSON, and start HTTP server'
    )
    parser.add_argument('--limit', type=int, help='Limit number of images to convert')
    parser.add_argument('--input', default='export_n1-obs_DataForGood.csv', help='Input CSV file')
    parser.add_argument('--output', default='labelstudio_tasks.json', help='Output JSON file')
    parser.add_argument('--force', action='store_true', help='Force re-download of existing images')

    args = parser.parse_args()

    # Download images and create JSON
    convert_csv_to_labelstudio(args.input, args.output, args.limit, args.force)

    # Start CORS-enabled HTTP server
    print("\n" + "="*60)
    print("Starting CORS-enabled HTTP server on http://localhost:8000")
    print("Import labelstudio_tasks.json in Label Studio UI")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    try:
        subprocess.run(['python3', 'cors_server.py'])
    except KeyboardInterrupt:
        print("\nHTTP server stopped.")
