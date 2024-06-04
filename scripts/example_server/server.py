from flask import Flask, render_template_string, jsonify
import json
import markdown
from markupsafe import escape
import fsspec

app = Flask(__name__)

# Function to render content based on type
def render_content(content):
    if content['type'] == 'text':
        return '<p>{}</p>'.format(escape(content['text']))
    elif content['type'] == 'html':
        # Wrap HTML content in an iframe
        iframe_content = f'<iframe srcdoc="{escape(content["text"])}" style="width:100%; height:500px; border:none;"></iframe>'
        return iframe_content
    elif content['type'] == 'md':
        return markdown.markdown(content['text'])
    else:
        return '<p>Unsupported content type</p>'

# Route to display the content
@app.route('/content/<path:dataname>/<int:idx>', methods=['GET'])
def display_content(dataname, idx):
    filename = f"gcs://marin-data/examples/{dataname}.jsonl"
    try:
        i = 0
        # fs = fsspec.filesystem("gcs")
        with fsspec.open(filename, 'r') as file:
            for line in file:
                if i < idx:
                    i += 1
                    continue
                record = json.loads(line.strip())
                idx = record['id']
                break
            else:
                return jsonify({'error': 'Record not found'}), 404
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

    print(f"Displaying content for {dataname} with ID: {idx}")
    rendered_content = [
        {'title': entry['title'], 'rendered': render_content(entry)}
        for entry in record['content']
    ]

    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Example with ID: {}, in {}</title>
        <style>
            .dropdown-header {{
                background-color: #f2f2f2;
                padding: 10px;
                margin: 0;
                cursor: pointer;
            }}
            .dropdown-content {{
                display: none;
                padding: 10px;
            }}
            .dropdown-content.show {{
                display: block;
            }}
        </style>
        <script>
            function toggleContent(id) {{
                var content = document.getElementById(id);
                content.classList.toggle("show");
            }}
        </script>
    </head>
    <body>
        <h1>Example with ID: {}, in {}</h1>
        <p> Each title is a dropdown, you can click on it</p>
    '''.format(idx, dataname, escape(idx), dataname)

    for i, item in enumerate(rendered_content):
        html_content += '''
        <div class="dropdown">
            <h2 class="dropdown-header" onclick="toggleContent('content-{}')">{}</h2>
            <div id="content-{}" class="dropdown-content">{}</div>
        </div>
        '''.format(i, escape(item['title']), i, item['rendered'])

    html_content += '''
    </body>
    </html>
    '''

    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True)
