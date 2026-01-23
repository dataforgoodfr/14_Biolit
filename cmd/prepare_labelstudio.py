import csv
import json
import argparse
import urllib.request
import os
import time
import subprocess
import socket
from pathlib import Path


def get_lan_ip():
    """Get LAN IP address, preferring ethernet over WiFi."""
    try:
        # Parse ip addr output to find interfaces
        result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
        lines = result.stdout.split('\n')

        interfaces = {}
        current_iface = None

        for line in lines:
            # Interface line (e.g., "2: eth0: <BROADCAST...")
            if ': ' in line and not line.startswith(' '):
                parts = line.split(': ')
                if len(parts) >= 2:
                    current_iface = parts[1].split('@')[0]
            # IPv4 address line
            elif 'inet ' in line and current_iface:
                ip = line.strip().split()[1].split('/')[0]
                if not ip.startswith('127.'):
                    interfaces[current_iface] = ip

        # Prefer ethernet (eth*, enp*, eno*) over wifi (wlan*, wlp*)
        for iface, ip in interfaces.items():
            if iface.startswith(('eth', 'enp', 'eno')):
                return ip

        # Fallback to wifi
        for iface, ip in interfaces.items():
            if iface.startswith(('wlan', 'wlp')):
                return ip

        # Fallback to any non-loopback
        if interfaces:
            return list(interfaces.values())[0]

    except Exception:
        pass

    # Last resort: use socket trick
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return 'localhost'

def convert_csv_to_labelstudio(csv_file, output_file, limit=None, force=False, host=None):
    tasks = []

    # Get host IP for image URLs
    if host is None:
        host = get_lan_ip()
    print(f"Using host IP: {host}")

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

            # Create task with annotation using LAN IP URL
            task = {
                "data": {
                    "captioning": f"http://{host}:8000/images/{filename}"
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
    parser.add_argument('--host', type=str, help='Override host IP for image URLs (default: auto-detect LAN IP)')

    args = parser.parse_args()

    # Get host IP
    host = args.host if args.host else get_lan_ip()

    # Download images and create JSON
    convert_csv_to_labelstudio(args.input, args.output, args.limit, args.force, host)

    # Start CORS-enabled HTTP server
    print("\n" + "="*60)
    print(f"Starting CORS-enabled HTTP server on http://{host}:8000")
    print("Import labelstudio_tasks.json in Label Studio UI")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")

    try:
        subprocess.run(['python3', 'cors_server.py', host])
    except KeyboardInterrupt:
        print("\nHTTP server stopped.")
