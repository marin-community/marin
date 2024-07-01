import os
import json
import re

def read_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_jsonl_file(data, output_file):
    with open(output_file, 'w') as file:
        for item in data:
            json_string = json.dumps(item)
            file.write(json_string + '\n')

def main():
    output_dir = 'output'
    html_dir = os.path.join(output_dir, 'html')
    md_dir = os.path.join(output_dir, 'markdown')
    output_file = 'server_convos.jsonl'

    jsonl_data = []

    for file_name in os.listdir(html_dir):
        if file_name.endswith('.html'):
            html_file_path = os.path.join(html_dir, file_name)
            md_file_name = file_name.replace('.html', '.md')
            md_file_path = os.path.join(md_dir, md_file_name)

            if os.path.exists(md_file_path):
                html_content = read_file_content(html_file_path)
                md_content = read_file_content(md_file_path)

                unique_id = re.findall(r'_(\d+)\.', file_name)[0]

                json_object = {
                    'id': unique_id,
                    'content': [
                        {
                            'title': 'Markdown',
                            'type': 'md',
                            'text': md_content
                        },
                        {
                            'title': 'HTML',
                            'type': 'html',
                            'text': html_content
                        }
                    ]
                }

                jsonl_data.append(json_object)

    write_jsonl_file(jsonl_data, output_file)
    print(f"JSONL file '{output_file}' created successfully.")

if __name__ == '__main__':
    main()