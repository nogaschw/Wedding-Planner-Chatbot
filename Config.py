class Config:
    def __init__(self):
        self.text_file_location = r'C:\Users\User\Documents\Wedplanner\TxtFiles'
        self.dataset = r'C:\Users\User\Documents\Wedplanner\dataset.csv'
        self.vector_store_path = r'C:\Users\User\Documents\Wedplanner\whatsapp_chat_faiss_cpu'
        self.OPENAI_API_KEY = "openai api key"
        self.gen_model = 'gpt-4o-mini'
        self.embeded_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"