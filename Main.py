import gradio as gr
from RAG.Retrieval import *
from RAG.Generation import *
from RAG.VectorStore import *

def main():
    config = Config()
    v_store = VectorStore()

    if not os.path.exists(config.vector_store_path):
        v_store.create_vector()
    vectorstore = v_store.load_vector_store()
    retrieval = Retrieval(vectorstore)
    rag = RAG(retrieval)
    # --- Gradio UI ---
    ui = gr.ChatInterface(rag.rag_answer, title="RAG Chatbot")
    ui.launch()

if __name__ == "__main__":
    main()