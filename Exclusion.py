import pandas as pd
import glob
import os

def copy_exclusion_sheet():
    """
    Find the SQF Excel file using glob, read the 'Exclusion' sheet,
    and save it to 'Exclusion.xlsx'
    """
    
    # Use glob to search for the SQF Excel file
    pattern = "*Outil*Lissage*.xlsm"
    excel_files = glob.glob(pattern, recursive=True)
    
    if not excel_files:
        print("No SQF Excel file found matching the pattern.")
        return
    
    # Use the first matching file
    source_file = excel_files[0]
    print(f"Found source file: {source_file}")
    
    try:
        # Read the 'Exclusion' sheet
        exclusion_data = pd.read_excel(source_file, sheet_name='Exclusion')
        print(f"Successfully read 'Exclusion' sheet with {len(exclusion_data)} rows")
        
        # Get the directory of the source file to save the output in the same location
        output_dir = os.path.dirname(source_file)
        output_file = os.path.join(output_dir, 'Exclusion.xlsx')
        
        # Save to Exclusion.xlsx
        exclusion_data.to_excel(output_file, sheet_name='Exclusion', index=False)
        print(f"Successfully saved to: {output_file}")
        
        return exclusion_data
        
    except Exception as e:
        print(f"Error processing the file: {str(e)}")
        return None

if __name__ == "__main__":
    copy_exclusion_sheet()