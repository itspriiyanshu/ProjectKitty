import os
import requests
import m3u8
import subprocess
from urllib.parse import urljoin, urlparse

# List of your Brightcove m3u8 URLs
m3u8_urls =[
    # "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/6725d065-f4f4-49f9-bac0-6666dff04378/10s/master.m3u8?fastly_token=NjgyNTU4NDJfNjI0ZTA2NmM0Y2EwZGViNzExZjQ5YWU1MjViNmY4YTZmZDQxOWE4MmI2Yjk1MWM1NzhmNGU5YTExODM3NjBjOA=="
    # "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/de706025-a394-4e94-afbf-4166ff7a21d1/10s/master.m3u8?fastly_token=NjgyNTU4NmJfYjdmZjg1MzA0ZjMzYWIxYzAyOWY5MGI4MWQ0ZDhkNzc0ZTNiYzZiOTVmOWY0M2M3MWFjZDU0ZmRiZjY4MjZiNQ=="
    # "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/56214f55-51cc-43fa-93da-04f3dea1a7fb/10s/master.m3u8?fastly_token=NjgyNTVjY2JfZjY1NWY0NWIzNGU0MTM1YjJhNmJiYzAwZTdiMDMwYjkyYTE4NWQ1NDY3NTNhOGQ4YjE4OTI4MGNlMTkzZTM5MA=="
    # "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/0fe488c9-d1a2-47ce-8418-c7a86a72179e/10s/master.m3u8?fastly_token=NjgyNTVkNDNfNDI3YmExOTJhNTcyZDkzZjBmNzc1NThhZDkxMjM1N2MxOGMwN2FhNTIzN2Q4ZTEwMmE0Zjg0NGJhNTM4NTNkZQ=="
    # "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/41cac1b9-90ed-4b3d-868a-3fb9aa1fb993/10s/master.m3u8?fastly_token=NjgyNTVkOTNfYjVjYjgxMGFjOGY3NWU1NDY1ZWU0ZWU2NDY1MDVlZGVkNWE3ODc4NzliYmNlN2Y3NWI3YzE2N2E1MTU3N2I2Yw=="
    # "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/63038174-d031-44e2-9924-0081d8c22d16/10s/master.m3u8?fastly_token=NjgyNjY0YjRfZmM5OTZhYjM4NjY3ODViYmNkM2YzOWZhN2RjMTA5MTYzMDg1MjI0ODA4NjI4ZmJkOGE1NjgxYjE3NmNkNGE5Yw=="
    # "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/a9057f50-80a1-47bf-bc41-3530e55ff57e/10s/master.m3u8?fastly_token=NjgyNjY1OGNfODdlNDRiZDM2ZDQ0MTUxZGU5YTI4M2IyMzk2Yzg3NjMyODFhODUxZjY2YzljMDc2ZjEwZTY4YTAyZmJjOWIzNA=="
    "https://manifest.prod.boltdns.net/manifest/v1/hls/v4/aes128/3588749423001/148cfa85-6735-403d-92b7-201c6b1bc299/10s/master.m3u8?fastly_token=NjgyNmUyYTNfMzk4M2RkNDg2ZThiZGExNDZhMWJkZTE0NzdkZmRhZDdlYjQxMGUxMTgwYzBkZGExZTZlNjg1NmRjYzFiNjUxMw=="
]

# Ensure output folder
os.makedirs('data', exist_ok=True)

def download_and_convert(m3u8_url, index):
    print(f"[+] Processing video {index + 1}...")

    headers = {'User-Agent': 'Mozilla/5.0'}

    # Load the master playlist
    master_resp = requests.get(m3u8_url, headers=headers)
    master_playlist = m3u8.loads(master_resp.text)

    if not master_playlist.is_variant:
        raise ValueError("Expected master playlist, but got media playlist.")

    # Select 1280x720 resolution stream
    target_stream = None
    for playlist in master_playlist.playlists:
        if playlist.stream_info.resolution == (1280, 720):
            target_stream = playlist
            break

    if not target_stream:
        raise ValueError("1280x720 stream not found.")

    stream_url = urljoin(m3u8_url, target_stream.uri)

    # Load the stream playlist
    stream_resp = requests.get(stream_url, headers=headers)
    stream_playlist = m3u8.loads(stream_resp.text)

    ts_dir = f"temp_ts_{index}"
    os.makedirs(ts_dir, exist_ok=True)

    # Handle decryption if needed
    key_data = None
    if stream_playlist.keys and stream_playlist.keys[0]:
        key = stream_playlist.keys[0]
        key_uri = urljoin(stream_url, key.uri)
        key_data = requests.get(key_uri, headers=headers).content

    ts_files = []
    for i, segment in enumerate(stream_playlist.segments):
        ts_url = urljoin(stream_url, segment.uri)
        ts_path = os.path.join(ts_dir, f"seg_{i:04d}.ts")

        ts_data = requests.get(ts_url, headers=headers).content

        # Decrypt if needed
        if key_data:
            from Crypto.Cipher import AES
            from Crypto.Util.Padding import unpad
            iv = bytes.fromhex(segment.key.iv.replace("0x", "")) if segment.key.iv else i.to_bytes(16, 'big')
            cipher = AES.new(key_data, AES.MODE_CBC, iv)
            ts_data = cipher.decrypt(ts_data)

        with open(ts_path, "wb") as f:
            f.write(ts_data)

        ts_files.append(ts_path)

    # Concatenate using ffmpeg
    file_list_path = os.path.join(ts_dir, "file_list.txt")
    with open(file_list_path, "w") as f:
        for file in ts_files:
            abs_path = os.path.abspath(file)
            f.write(f"file '{abs_path}'\n")

    output_file = os.path.join("data", f"video_{index+1}.mp4")
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", file_list_path,
        "-c", "copy", output_file, "-y"
    ]
    subprocess.run(cmd, check=True)

    print(f"[✓] Saved: {output_file}")

    # Cleanup
    for file in ts_files + [file_list_path]:
        os.remove(file)
    os.rmdir(ts_dir)

def main():
    for idx, url in enumerate(m3u8_urls):
        try:
            download_and_convert(url, idx)
        except Exception as e:
            print(f"[!] Error processing URL {url[:60]}...: {e}")

if __name__ == "__main__":
    main()
