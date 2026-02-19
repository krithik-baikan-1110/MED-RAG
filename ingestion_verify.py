import weaviate

client = weaviate.Client("http://localhost:8080")
schema = client.schema.get()

for c in schema["classes"]:
    if c["class"] == "MedicalReport":
        print("\nClass:", c["class"])
        for prop in c["properties"]:
            print(f" - {prop['name']} ({prop['dataType']})")
query = (
    client.query
    .get("MedicalReport", ["image_path", "report_text"])
    .with_limit(3)
    .do()
)
print(query)
count = client.query.aggregate("MedicalReport").with_meta_count().do()
print("Total stored objects:", count["data"]["Aggregate"]["MedicalReport"][0]["meta"]["count"])
