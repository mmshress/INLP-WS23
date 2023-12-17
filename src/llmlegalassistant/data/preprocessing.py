'''
Regex-based cleanup for the text files before pushing into Elasticsearch
'''
import os
import re

class Preprocessor():
    def __init__(self) -> None:
        REGEXES = [r"^.*\@import.*$", r"^xml version.*$", r"^.*\.xml$", r"^.*\$\(document\)\.ready.*\;$"]
        self.REGEXES = [re.compile(reggie) for reggie in REGEXES]
    
    # We are only using the text files for pushing into Elasticsearch
    def clean_text_files(self, text_files_dir: str) -> None:
        text_files = [f for f in os.listdir(text_files_dir) if os.path.isfile(os.path.join(text_files_dir, f))]
        for file_path in text_files:
            with open(text_files_dir + file_path, "r") as text_file:
                lines = []
                for line in text_file.readlines():
                    for reggie in self.REGEXES: 
                        if not reggie.match(line) or not line.strip() == "":
                            lines.append(line)
            with open(text_files_dir + file_path, "w") as text_file:
                text_file.writelines(lines)
        
