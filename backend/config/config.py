class Config:
    class Dirs:
        JSON_DIR = "json"
        PDF_DIR = "pdf"
        RESULT_DIR = "result"
        PROGRESS_DIR = "progress"
        PROGRESS_FILE = "progress.json"

    class Model:
        NAME = "gemma3:12b"
        TEMPERATURE = 0.1
        TOP_P = 0.9
        NUM_CTX = 65536    # overkill but just in case
        NUM_PREDICT = -1