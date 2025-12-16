import re
from datetime import datetime

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)")
DATE_ONLY_REGEX = re.compile(r"[\d./\\-]{3,10}")

WHATSAPP_PATTERN = re.compile(
    r"\[(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}:\d{2})\] ~\s*(.*?): (.*)"
)

def clean_text(msg: str) -> str | None:
    """Remove URLs + basic noise. Return cleaned text or None if noise."""
    # Remove URLs
    msg = URL_REGEX.sub("", msg).strip()
    msg = msg.replace("\u200e<This message was edited>", "")

    # 1) Empty after cleaning
    if not msg:
        return None

    # 2) Emoji-only or no alphanumeric (e.g., "😂😂", "😍")
    if all(not ch.isalnum() for ch in msg):
        return None

    # 3) WhatsApp system messages (English metadata)
    if any(x in msg for x in ["invited", "created", "changed", "left", "joined", "deleted", "added", "removed"]):
        return None

    # 4) Date-only messages: "21/3", "03/07/24", "16.08", "2024-06-21"
    if DATE_ONLY_REGEX.fullmatch(msg):
        return None

    # 5) Pure numbers or one letter
    if msg.isdigit() or len(msg.strip()) == 1:
        return None

    # 6) one not meaningful word
    words = msg.split()
    if len(words) == 1 and any(x in msg for x in ["חח", "תודה"]):
        return None 

    return msg

def parse_and_merge_messages(raw_lines, merge_seconds=40):
    """
    Parse WhatsApp raw export lines, clean each message,
    merge consecutive messages from the same sender,
    and return a final clean message list.
    """

    parsed_msgs = []

    # 1) Parse raw lines
    for line in raw_lines:
        m = WHATSAPP_PATTERN.match(line)
        if not m:
            continue  # skip lines that don't match WhatsApp format

        date_str, time_str, name, text = m.groups()
        dt = datetime.strptime(date_str + " " + time_str, "%d/%m/%Y %H:%M:%S")

        cleaned = clean_text(text)
        if cleaned is None:
            continue  # skip noise

        parsed_msgs.append({
            "time": dt,
            "name": name,
            "text": cleaned
        })

    # 2) Merge consecutive messages from the same sender
    merged = []
    for msg in parsed_msgs:
        if not merged:
            merged.append(msg)
            continue

        last = merged[-1]

        same_sender = msg["name"] == last["name"]
        close_in_time = (msg["time"] - last["time"]).total_seconds() <= merge_seconds

        if same_sender and close_in_time:
            # merge text and update time
            last["text"] += "\n" + msg["text"]
            last["time"] = msg["time"]
        else:
            merged.append(msg)

    return merged