from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="varunvaddi/google-ads-rag-embeddings",
    repo_type="dataset",
    exist_ok=True
)

for filename in ["bm25.pkl", "embeddings.npy", "faiss.index", "metadata.json"]:
    api.upload_file(
        path_or_fileobj=f"data/embeddings/{filename}",
        path_in_repo=filename,
        repo_id="varunvaddi/google-ads-rag-embeddings",
        repo_type="dataset",
    )
    print(f"✅ Uploaded {filename}")