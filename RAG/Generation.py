from Agent.Agent import *
from Agent.Prompts import rag_generative_prompt

TOPIC_KEYWORDS = {
    "venue": ["אולם", "גן", "מקום"],
    "photographer": ["צלם", "צילום"],
    "video": ["וידאו", "סטורי"],
    "dj": ["די ג׳יי", "dj", "מוזיקה", "דיי-גיים", "דיי-גיי"],
    "live_music": ["להקה", "מוזיקה חיה", "סקספונ", "מתופף"],
    "makeup_hair": ["איפור", "שיער", "מאפרת"],
    "dress": ["שמלה", "שמלות", "אוברול"],
    "suit": ["חליפה"],
    "shoes": ["נעליים"],
    "magnets": ["מגנטים"],
    "social_media": ["אינסטגרם", "טיקטוק", "סטורי"],
    "invitation": ["הזמנה", "הזמנות"],
    "beauty_prep": ["טיפוח", "סקין"],
    "getting_ready": ["התארגנות"],
    "manager": ["מנהל אירוע"],
    "rsvp": ["אישורי הגעה", "RSVP"],
    "decor": ["עיצוב", "פרחים"],
    "bar": ["בר", "אלכוהול"],
    "food": ["אוכל", "קייטרינג"],
    "cake": ["עוגה"],
    "guests": ["אורחים"],
    "tables": ["שולחנות", "סידור ישיבה"],
    "rabbinate_halacha": ["רבנות", "הלכה", "הפרשת כלה"],
    "attractions": ["אטרקציות"],
    "jewelry": ["טבעות", "תכשיטים"],
    "bachelorette": ["רווקות"],
    "henna": ["חינה"],
    "shabbat_hatan": ["שבת חתן"],
    "engagement": ["אירוסין", "הצעה"],
}

class RAG:
    def __init__(self, retrieval):
        self.get_retriever = retrieval.get_retriever

    def context_prepration(self, docs):
        return "\n\n---\n\n".join(
            d.page_content
            for d in docs
        ) if docs else "אין נתונים רלוונטיים שנשלפו."

    def dedupe_docs(self, primary, secondary):
        primary_ids = {d.id for d in primary}
        return [d for d in secondary if d.id not in primary_ids]

    def infer_topic(self, query):
        for topic, kws in TOPIC_KEYWORDS.items():
            if any(k in query for k in kws):
                return topic
        return None
    

    def rag_answer(self, query: str, history: str = "") -> str:
        # 1. Topic Retrieval
        topic_name = self.infer_topic(query) 
        
        # 2. Document Retrieval
        docs_topics = self.get_retriever(topic=topic_name).invoke(query)
        docs_general = self.dedupe_docs(self.get_retriever().invoke(query), docs_topics)
        
        # 3. Prompt Formatting (using the RAG_GENERATIVE_PROMPT)
        final_prompt_input = rag_generative_prompt.format(
            user_query=query,
            context_topic=self.context_prepration(docs_topics),
            context_general=self.context_prepration(docs_general)
        )
        
        # 4. Generation
        return ask_agent(prompt=final_prompt_input, data=query)