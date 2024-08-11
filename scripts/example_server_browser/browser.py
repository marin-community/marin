import json
import os
import traceback
from collections import defaultdict

import comrak_py
import fsspec
from flask import render_template_string, jsonify, Flask, request, url_for
from markupsafe import escape

from marin.utils import get_gcs_path

app = Flask(__name__)


def get_format_type(format_type):
    '''Given format_type from jsonl file, get rendering format'''
    if format_type == 'md' or format_type == 'markdown':
        return 'markdown'
    elif format_type == 'html' or format_type == 'xml':
        return 'html'
    elif format_type == 'txt' or format_type == 'text':
        return 'text'
    else:  # We support the only three types above
        raise ValueError(f'Unsupported format type: {format_type}')


@app.route('/')
def list_files():
    path = request.args.get('path', '')
    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    if path:
        files = fs.ls(f"gcs://{path}")
    else:
        return jsonify({'error': 'Path parameter is required'}), 400

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Marin Data Browser</title>
    </head>
    <body>
    <h1>Marin Data Browser</h1>
    <ul>
    """

    parent_path = os.path.dirname(path) if path else ''
    if parent_path:
        html_content += f'<li><a href="{url_for("list_files", path=parent_path)}">.. (Parent)</a></li>'

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
    '''Render content based on format type'''

    content_render = ""
    if format_type == 'text':
        content_render = '<p style="white-space: pre-line;">{}</p>'.format(escape(content))
    elif format_type == 'html':
        content_render = f'<iframe srcdoc="{escape(content)}" style="width:100%; height:500px; border:none;"></iframe>'
    elif format_type == 'markdown':
        content_render = comrak_py.gfm_to_html(content)
    else:
        raise ValueError(f'Unsupported format type: {format_type}')

    return content_render


def render(path, record, title_suffix=""):
    '''Render the title and content of the record'''

    # If "format" is not present then we render it as text
    rendered_content = []
    format_type = get_format_type(record.get('format', 'text'))
    rendered_content.append({'title': f"{path} ; {format_type} ; {title_suffix}",
                             'rendered': render_content(record['text'], format_type)})

    if format_type == 'markdown':  # We also render raw markdown
        rendered_content.append({'title': f"{path} ; Raw Markdown ; {title_suffix}",
                                 'rendered': render_content(record['text'], 'text')})

    return rendered_content


# Route to display the content
@app.route('/content', methods=['GET'])
def display_content():
    paths = request.args.get('paths')
    index = int(request.args.get('index', 0))
    count = int(request.args.get('count', 1))  # Get the count parameter with a default value of 1

    if not paths:
        return jsonify({'error': 'Paths parameter is required'}), 400

    paths = paths.split(',')
    rendered_content = []
    jsons_to_render = defaultdict(list)

    try:
        for path in paths:
            filename = get_gcs_path(path)
            with fsspec.open(filename, 'rt', compression='gzip', use_listings_cache=False) as file:
                for i, line in enumerate(file):
                    if i >= index and i < index + count:
                        record = json.loads(line.strip())
                        jsons_to_render[i].append((record, path))
                        # rendered_content += render(path, record, title_suffix)
                    if i >= index + count - 1:  # Stop reading after we've collected `count` items
                        break

        for i, records in jsons_to_render.items():
            for record, path in records:
                rendered_content += render(path, record, f"Index: {i}")

        html_content = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Records {} to {} of {}</title>
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
            <h1>Example Records from Index {} to {} in {}</h1>
            <p> Each title is a dropdown, you can click on it</p>
        '''.format(index, index + count - 1, ', '.join(paths), index, index + count - 1, ', '.join(paths))

        for i, item in enumerate(rendered_content):
            html_content += '''
            <div class="dropdown">
                <h2 class="dropdown-header" onclick="toggleContent('content-{}')">{}</h2>
                <div id="content-{}" class="dropdown-content"> {{% raw %}} {} {{% endraw %}}</div>
            </div>
            '''.format(i, escape(item['title']), i, item['rendered'])

        if index > 0:
            html_content += f'<a href="/content?paths={",".join(paths)}&index={0}&count={count}">First</a>'
            html_content += ' | '
            html_content += f'<a href="/content?paths={",".join(paths)}&index={max(0, index - count)}&count={count}">Previous</a>'
            html_content += ' | '
        html_content += f'<a href="/content?paths={",".join(paths)}&index={index + count}&count={count}">Next</a>'

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
    app.run()
