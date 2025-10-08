import pandas as pd

def process_groundwater_excel(input_path='CentralReport1759351660554.xlsx',
                             output_path='cleaned_groundwater_data.csv'):
    # Load raw Excel file, skip first 7 rows (based on file analysis)
    df = pd.read_excel(input_path, skiprows=7)
    
    # Columns to keep (adjusted for spacing/actual headers)
    keep_cols = [
        'STATE',
        'DISTRICT',
        'ASSESSMENT UNIT',
        'Rainfall (mm)',
        'Total Ground Water Availability in the area (ham)',
        'Stage of Ground Water Extraction (%)',
        'Ground Water Extraction for all uses (ha.m)'
    ]
    
    # Find actual matching columns regardless of exact header string
    found_cols = {col: next((c for c in df.columns if col in str(c)), None) for col in keep_cols}
    cols_to_use = [found_cols[c] for c in found_cols if found_cols[c] is not None]
    
    # Filter and rename
    df_clean = df[cols_to_use].copy()
    df_clean = df_clean.dropna(subset=[found_cols['DISTRICT']])
    df_clean.columns = [
        'State', 'District', 'AssessmentUnit', 'Rainfall_mm',
        'GroundWaterAvailability_ham', 'ExtractionStage_Percent',
        'ExtractionVolume_ha_m'
    ]
    
    # Save the cleaned CSV, making sure it overwrites the previous version
    df_clean.to_csv(output_path, index=False)
    print(f'[INFO] Cleaned CSV saved to {output_path}')

if __name__ == "__main__":
    process_groundwater_excel()
