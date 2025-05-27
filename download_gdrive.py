import os
import io
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import json

def setup_drive_service():
    """Setup Google Drive service using service account credentials"""
    # Service account info - you'll need to set this as an environment variable
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
        # Get file metadata
        file_metadata = service.files().get(fileId=file_id).execute()
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

def list_and_download_folder_contents(service, folder_id, destination_folder='.'):
    """List and download all files in a Google Drive folder"""
    try:
        # Create destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # List all files in the folder
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print(f"No files found in folder {folder_id}")
            return
        
        print(f"Found {len(files)} files in the folder:")
        
        downloaded_count = 0
        for file in files:
            file_id = file['id']
            file_name = file['name']
            mime_type = file['mimeType']
            
            print(f"\n- {file_name} (Type: {mime_type})")
            
            # Skip Google Drive native formats (Docs, Sheets, etc.) for now
            if mime_type.startswith('application/vnd.google-apps'):
                print(f"  Skipping Google native format: {mime_type}")
                continue
            
            # Download the file
            if download_file(service, file_id, file_name, destination_folder):
                downloaded_count += 1
        
        print(f"\nDownload complete! Downloaded {downloaded_count} files to '{destination_folder}'")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in os.listdir(destination_folder):
            file_path = os.path.join(destination_folder, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
                
    except Exception as e:
        print(f"Error accessing folder: {str(e)}")

def main():
    """Main function to download all files from Google Drive folder"""
    # Your Google Drive folder ID
    FOLDER_ID = "1HwQYHUTF33-qQc8CnPQu2l1qOKQGThFO"
    DOWNLOAD_FOLDER = "gdrive_downloads"
    
    print("Setting up Google Drive service...")
    service = setup_drive_service()
    
    print(f"Downloading contents from folder ID: {FOLDER_ID}")
    list_and_download_folder_contents(service, FOLDER_ID, DOWNLOAD_FOLDER)

if __name__ == "__main__":
    main()
