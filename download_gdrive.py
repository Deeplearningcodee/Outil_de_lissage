import os
import io
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import json

def setup_drive_service():
    """Setup Google Drive service using service account credentials"""
    service_account_info = json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
    
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    
    service = build('drive', 'v3', credentials=credentials)
    return service

def download_file(service, file_id, file_name, destination_folder='.'):
    """Download a file from Google Drive"""
    try:
        print(f"Downloading: {file_name}")
        
        # Create destination path
        file_path = os.path.join(destination_folder, file_name)
        
        # Request file content
        request = service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download progress: {int(status.progress() * 100)}%")
        
        # Write file to disk
        with open(file_path, 'wb') as f:
            f.write(file_io.getvalue())
        
        print(f"Successfully downloaded: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading {file_name}: {str(e)}")
        return False

def download_folder_recursively(service, folder_id, destination_folder='.', folder_name=None):
    """Recursively download all files and subfolders from a Google Drive folder"""
    try:
        # Create destination folder if it doesn't exist
        if folder_name:
            destination_folder = os.path.join(destination_folder, folder_name)
        os.makedirs(destination_folder, exist_ok=True)
        
        print(f"\nProcessing folder: {destination_folder}")
        
        # List all files and folders in the current folder
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print(f"No files found in folder {folder_id}")
            return 0
        
        print(f"Found {len(files)} items in this folder")
        
        downloaded_count = 0
        for file in files:
            file_id = file['id']
            file_name = file['name']
            mime_type = file['mimeType']
            
            print(f"\n- {file_name} (Type: {mime_type})")
            
            # If it's a folder, recurse into it
            if mime_type == 'application/vnd.google-apps.folder':
                print(f"  Entering subfolder: {file_name}")
                subfolder_count = download_folder_recursively(
                    service, file_id, destination_folder, file_name
                )
                downloaded_count += subfolder_count
            
            # Skip other Google Drive native formats (Docs, Sheets, etc.)
            elif mime_type.startswith('application/vnd.google-apps'):
                print(f"  Skipping Google native format: {mime_type}")
                continue
            
            # Download regular files
            else:
                if download_file(service, file_id, file_name, destination_folder):
                    downloaded_count += 1
        
        return downloaded_count
        
    except Exception as e:
        print(f"Error accessing folder {folder_id}: {str(e)}")
        return 0

def list_all_downloaded_files(root_folder):
    """List all downloaded files recursively"""
    print(f"\nAll downloaded files in '{root_folder}':")
    total_files = 0
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            relative_path = os.path.relpath(file_path, root_folder)
            print(f"  - {relative_path} ({size} bytes)")
            total_files += 1
    print(f"\nTotal files downloaded: {total_files}")

def main():
    """Main function to download all files from Google Drive folder"""
    # Your Google Drive folder ID
    FOLDER_ID = "1HwQYHUTF33-qQc8CnPQu2l1qOKQGThFO"
    DOWNLOAD_FOLDER = "gdrive_downloads"
    
    print("Setting up Google Drive service...")
    service = setup_drive_service()
    
    print(f"Downloading contents from folder ID: {FOLDER_ID}")
    downloaded_count = download_folder_recursively(service, FOLDER_ID, DOWNLOAD_FOLDER)
    
    print(f"\nDownload complete! Downloaded {downloaded_count} files total.")
    
    # List all downloaded files
    list_all_downloaded_files(DOWNLOAD_FOLDER)

if __name__ == "__main__":
    main()
