import os
import json
from Config import *
from Agent.Agent import *
from Agent.Prompts import *
from datetime import timedelta
from collections import defaultdict
from Preprocess.Create_Data import *

def chunk_list(data, chunk_size):
    """Yield successive n-sized chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def process_all_messages(all_raw_messages, chunk_size=100):
    """
    Main pipeline function to process all messages in chunks.
    Assumes you have a function called 'classify_messages_with_llm'
    that takes a list of messages and returns the categorized list.
    """
    classified_results = []
    
    # Iterate through chunks
    for i, chunk in enumerate(chunk_list(all_raw_messages, chunk_size)):
        print(f"Processing chunk {i + 1}...")
        
         # 1. Prepare the text block with IDs
        message_block = ""
        for idx, msg in enumerate(chunk):
            # We include the name because context matters (e.g., Maya asking vs Maya answering)
            message_block += f"{idx}: {msg['name']}: {msg['text']}\n"
        
        try:
            # 2. Call the LLM with the SYSTEM_PROMPT and the message block
            response = ask_agent(wedding_topics_classifier, message_block)

            # 3. Robust Parsing and Mapping (as defined in our previous fixes)
            clean_text = response.replace("```json", "").replace("```", "").strip()
            results = json.loads(clean_text)
            
            # Map categories back safely
            clean_results = {str(k).replace('"', ''): v for k, v in results.items()}
            
            for j, msg in enumerate(chunk):
                # Use str(j) because the keys in the JSON are string IDs (0, 1, 2...)
                msg['topic'] = clean_results.get(str(j), "unknown") 
                
            classified_results.extend(chunk)

        except Exception as e:
            print(f"Error processing chunk {i + 1}. Failed to parse JSON or call model: {e}")
            # Fallback: If classification fails, tag all messages in the chunk as 'unknown'
            for msg in chunk:
                msg['topic'] = "unknown"
            classified_results.extend(chunk)
            
    return classified_results

def aggregate_topic_chunks(messages, time_window_minutes=60):
    """
    Groups messages by topic and then concatenates consecutive messages
    within a specific time window into cohesive RAG chunks.
    """
    topic_chunks = defaultdict(list)
    
    # Sort messages by time to ensure proper sequence
    messages.sort(key=lambda x: x['time'])

    for msg in messages:
        topic = msg['topic']
        if topic in ['general_chat', 'unknown']:
            continue # Skip non-actionable topics

        # Prepare the current message string for concatenation
        message_str = f"{msg['text']}"
        
        if not topic_chunks[topic]:
            # Start a new chunk if the topic list is empty
            topic_chunks[topic].append({
                'topic': topic,
                'messages': [message_str],
                'start_time': msg['time']
            })
        else:
            last_chunk = topic_chunks[topic][-1]
            time_diff = msg['time'] - last_chunk['start_time']
            
            # Check if the message is close to the last one (within the time window)
            # and that it is the same topic (implied by the defaultdict key)
            if time_diff < timedelta(minutes=time_window_minutes):
                last_chunk['messages'].append(message_str)
            else:
                # Start a new chunk for the same topic (but different time chapter)
                topic_chunks[topic].append({
                    'topic': topic,
                    'messages': [message_str],
                    'start_time': msg['time']
                })
                
    # Flatten the result into a list of RAG documents (Topic, Raw Text)
    rag_documents = []
    for topic, chunks in topic_chunks.items():
        for chunk in chunks:
            raw_text = "[MES]".join(chunk['messages'])
            rag_documents.append({
                'topic': topic,
                'raw_text': raw_text,
                'timing': chunk['start_time']
            })

    return rag_documents


def synthesize_data(aggregated_chunks, batch_size=30):
    """
    Runs the LLM synthesis on each aggregated chunk to create the final dataset.
    """
    final_rag_dataset = []
    errors = []
    num_chunks = len(aggregated_chunks)
    
    for start_index in range(0, num_chunks, batch_size):
        end_index = min(start_index + batch_size, num_chunks)
        print(f"\n--- Processing Batch {start_index // batch_size + 1} (Chunks {start_index} to {end_index-1}) ---")       
        complit_chunk = "" 
        for i in range(start_index, end_index):
            chunk = aggregated_chunks[i]
            complit_chunk += f"id: {i}, Topic: {chunk['topic']}, Messages: {chunk['raw_text']}\n"
        try:
            response = ask_agent(summary_prompt, complit_chunk)
            # 3. Robust Parsing
            clean_text = response.replace("```json", "").replace("```", "").strip()
            synthesized_data = json.loads(clean_text)
            
            # 4. Final Data Structure for Vector DB
            for i, row in synthesized_data.items():
                curr_chunk = aggregated_chunks[int(i)]
                final_rag_dataset.append({
                    "source_topic": curr_chunk['topic'],
                    "summary_text": row['summary'], # This is your vector content
                    "all_names": row['all_names'], # Filterable fields
                    "locations": row['locations'],
                    "original_msg": curr_chunk['raw_text'], 
                    "timing": curr_chunk['timing']
                })

        except Exception as e:
            print(f"Error during synthesis of chunk {i + 1}: {e}")
            errors.append(clean_text)
            # Skip or log error, ensuring the process continues

    return final_rag_dataset, errors

def full_process():
    config = Config()
    files = os.listdir(config.text_file_location)

    rag_dataset = []
    
    for file in files:
        message = []
        with open(os.path.join(config.text_file_location, file),"r", encoding="utf-8") as f:
            raw = f.readlines()
        messages.extend(parse_and_merge_messages(raw))

        classified_results = process_all_messages(messages)
        chunks = aggregate_topic_chunks(classified_results, time_window_minutes=120)
        rag_dataset_i, errors = synthesize_data(chunks)
        rag_dataset.extend(rag_dataset_i)

    pd.DataFrame(rag_dataset).to_csv(config.saved_path)

if __name__ == "__main__":
    full_process()