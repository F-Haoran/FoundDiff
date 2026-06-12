from idc_index import IDCClient
from data.paths import MAYO_DICOM_ROOT

print("Connecting to IDC Database...")
client = IDCClient()

# Query for chest CTs from the AAPM-Mayo dataset
query = """
SELECT collection_id, PatientID, SeriesInstanceUID 
FROM index 
WHERE collection_id = 'ldct_and_projection_data' AND Modality = 'CT'
LIMIT 5
"""

print("Searching for datasets...")
results = client.sql_query(query)

print("Starting download...")
client.download_dicom_series(
    seriesInstanceUID=results["SeriesInstanceUID"].tolist(), 
    downloadDir=MAYO_DICOM_ROOT
)
print(f"Download complete. Check {MAYO_DICOM_ROOT}")