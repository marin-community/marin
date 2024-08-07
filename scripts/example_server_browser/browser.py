import json
import traceback

import comrak_py
import fsspec
import os
from flask import render_template_string, jsonify, Flask, request, url_for
from markupsafe import escape

from marin.utils import get_gcs_path

app = Flask(__name__)


@app.route('/')
def list_files():
    path = request.args.get('path', '')
    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    if path:
        files = fs.ls(f"gcs://{path}")
    else:
        files = fs.ls("gcs://")

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>GCP File Browser</title>
    </head>
    <body>
    <h1>GCP File Browser</h1>
    <ul>
    """

    parent_path = os.path.dirname(path) if path else ''
    if parent_path:
        html_content += f'<li><a href="{url_for("list_files", path=parent_path)}">.. (Back)</a></li>'

    for file in files:
        file_name = file.split('/')[-1]
        if file.endswith('jsonl.gz'):
            html_content += f'<li><a href="{url_for("display_content", paths=file, index=0)}">{file_name}</a></li>'
        else:  # here we are assuming everything else is a directory. Using fs.isdir() is not reliable
            html_content += f'<li><a href="{url_for("list_files", path=file)}">{file_name}/</a></li>'

    html_content += """
    </ul>
    </body>
    </html>
    """

    return render_template_string(html_content)


def render_content(content, format_type):
    if format_type == 'text' or format_type == 'txt':
        return '<p>{}</p>'.format(escape(content))
    elif format_type == 'html' or format_type == 'xml':
        iframe_content = f'<iframe srcdoc="{escape(content)}" style="width:100%; height:500px; border:none;"></iframe>'
        return iframe_content
    elif format_type == 'md':
        return comrak_py.gfm_to_html(content)
    else:
        return '<p>Unsupported content type, Rendering as plain text <br> {}</p>'.format(escape(content))


# Route to display the content
@app.route('/content', methods=['GET'])
def display_content():
    paths = request.args.get('paths')
    idx = int(request.args.get('index', 0))

    if not paths:
        return jsonify({'error': 'Paths parameter is required'}), 400

    paths = paths.split(',')
    record = None
    try:
        rendered_content = []
        for path in paths:
            filename = get_gcs_path(path)
            with fsspec.open(filename, 'rt', compression='gzip', use_listings_cache=False) as file:
                for i, line in enumerate(file):
                    if i == idx:
                        record = json.loads(line.strip())
                        break

            if not record:
                return jsonify({'error': f'Record not found in {path} at index {idx}'}), 404

            source_id = record['id']

            if "format" not in record:
                record["format"] = "unsupported text"

            if record['format'] == 'md':
                html_content = ('<iframe srcdoc="{}" style="width:100%; height:500px; border:none;"></iframe>'.
                                format(escape(record["text"]).replace('\n', '<br>')))
                rendered_content.append({
                    'title': f"{path},raw md",
                    'rendered': render_content(html_content, "html")
                })

            rendered_content.append(
                {'title': f"{path}, {record['format']}", 'rendered': render_content(record["text"], record["format"])})

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
        '''.format(escape(source_id), ', '.join(paths), escape(source_id), ', '.join(paths))

        for i, item in enumerate(rendered_content):
            html_content += '''
            <div class="dropdown">
                <h2 class="dropdown-header" onclick="toggleContent('content-{}')">{}</h2>
                <div id="content-{}" class="dropdown-content"> {{% raw %}} {} {{% endraw %}}</div>
            </div>
            '''.format(i, escape(item['title']), i, item['rendered'])

        if int(idx) > 0:
            html_content += f'<a href="/content?paths={",".join(paths)}&index={0}">First</a>'
            html_content += ' | '
            html_content += f'<a href="/content?paths={",".join(paths)}&index={int(idx) - 1}">Previous</a>'
            html_content += ' | '
        html_content += f'<a href="/content?paths={",".join(paths)}&index={int(idx) + 1}">Next</a>'

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
    app.run(debug=True)
