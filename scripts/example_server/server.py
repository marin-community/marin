from flask import Flask, render_template_string, jsonify
import json
from markupsafe import escape
import fsspec
import grip
import os
import traceback

app = Flask(__name__)
githubRenderer = grip.GitHubRenderer()

# Function to render content based on type
def render_content(content):
    if content['type'] == 'text':
        return '<p>{}</p>'.format(escape(content['text']))
    elif content['type'] == 'html':
        # Wrap HTML content in an iframe
        iframe_content = f'<iframe srcdoc="{escape(content["text"])}" style="width:100%; height:500px; border:none;"></iframe>'
        return iframe_content
    elif content['type'] == 'md':
        return githubRenderer.render(content['text'])
    else:
        return '<p>Unsupported content type</p>'

# Route to display the content
@app.route('/', methods=['GET'])
def display_home():
    fs = fsspec.filesystem("gcs")
    files = fs.ls("gcs://marin-data/examples/")
    files = [file.split('/')[-1][:-6] for file in files]
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Home</title>
    </head>
    <body>
    <h1>Home</h1>"""
    for i, file in enumerate(files):
        # add links to files if they exist
        if file:
            html_content += f'''<li><a href="/content/{file}/0">{file}</a><br>'''
    html_content += '''
    </body>
    </html>
    '''
    return render_template_string(html_content)

# Route to display the content
@app.route('/content/<path:dataname>/<int:idx>', methods=['GET'])
def display_content(dataname, idx):
    filename = f"gcs://marin-data/examples/{dataname}.jsonl"
    try:
        i = 0
        with fsspec.open(filename, 'r') as file:
            for line in file:
                if i < idx:
                    i += 1
                    continue
                record = json.loads(line.strip())
                source_id = record['id']
                break
            else:
                return jsonify({'error': 'Record not found'}), 404


        print(f"Displaying content for {dataname} with index: {idx}, ID: {source_id}")
        for entry in record['content']:
            if entry['type'] == 'md':
                record['content'].append({'title': f"raw {entry['title']}", 'type': 'html', 'text': '<iframe srcdoc="{}" style="width:100%; height:500px; border:none;"></iframe>'.format(escape(entry["text"]).replace('\n', '<br>'))})
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
        '''.format(escape(source_id), dataname, escape(source_id), dataname)

        for i, item in enumerate(rendered_content):
            html_content += '''
            <div class="dropdown">
                <h2 class="dropdown-header" onclick="toggleContent('content-{}')">{}</h2>
                <div id="content-{}" class="dropdown-content">{}</div>
            </div>
            '''.format(i, escape(item['title']), i, item['rendered'])

        if int(idx) > 0:
            html_content += f'<a href="/content/{dataname}/{0}">First</a>'
            html_content += ' | '
            html_content += f'<a href="/content/{dataname}/{int(idx)-1}">Previous</a>'
            html_content += ' | '
        html_content += f'<a href="/content/{dataname}/{int(idx)+1}">Next</a>'

        html_content += '''
        </body>
        </html>
        '''

        return render_template_string(html_content)
    except Exception as e:
        stack = traceback.format_exc().replace("\n", "<br>")
        error_msg = f'''Server internal Error<br>Error: {e}<br>Stack Trace: {stack}'''
        return error_msg, 500

if __name__ == '__main__':
    # Check if the deployment variable is set (to run on the server)
    if 'DEPLOYMENT' in os.environ:
        app.run(host='0.0.0.0', port=80)
    else:
        app.run(debug=True)
