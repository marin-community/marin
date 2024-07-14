import json
import os
import traceback

import comrak_py
import google.auth
from flask import render_template_string, Blueprint
from google.cloud import storage
from markupsafe import escape

bp = Blueprint('v2', __name__)

# Initialize GCP Storage client
credentials, project = google.auth.default()
storage_client = storage.Client(credentials=credentials, project=project)


# Function to render content based on type
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


def list_gcs_dir(bucket_name, prefix):
    if prefix.endswith('/'):
        prefix = prefix[:-1]
    prefix = prefix + '/'
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter='/')

    temp_list = []
    for blob in blobs:  # We just need to do this to populate prefixes
        temp_list.append(blob.name)

    subdir = [blob.split("/")[-2] for blob in blobs.prefixes]
    return subdir


# Route to display the home page
@bp.route('/', methods=['GET'])
def display_home():
    bucket_name = "marin-data"
    prefix = "examples/"
    domains = list_gcs_dir(bucket_name, prefix)

    versions = {}
    for domain in domains:
        versions[domain] = list_gcs_dir(bucket_name, f"{prefix}{domain}/")

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Home</title>
    </head>
    <body>
    <h1>Home</h1>"""
    for domain in domains:
        for version in versions[domain]:
            html_content += f'''<li><a href="/v2/content/{domain}/{version}/0">{domain}, {version}</a><br>'''
    html_content += '''
    </body>
    </html>
    '''
    return render_template_string(html_content)


# Route to display the content
@bp.route('/content/<path:domain>/<path:version>/<int:idx>', methods=['GET'])
def display_content(domain, version, idx):
    try:
        content_data = {}
        bucket_name = "marin-data"
        base_path = f"examples/{domain}/{version}/"
        formats = list_gcs_dir(bucket_name, base_path)
        formats = [fmt for fmt in formats if fmt.lower().startswith(('xml', 'html', 'md', 'txt', 'text'))]

        bucket = storage_client.bucket(bucket_name)

        for fmt in formats:
            filename = f"{base_path}{fmt}/samples.jsonl"
            blob = bucket.get_blob(filename)
            blob_md5 = blob.md5_hash

            # Implement this (Todo: Abhinav Garg)
            # content = get_cached_content(bucket_name, filename, blob_md5)

            if not blob:
                continue

            content = blob.download_as_text().splitlines()

            if idx < len(content):
                record = json.loads(content[idx].strip())
                content_data[fmt] = record['text']
            else:
                content_data[fmt] = None

        rendered_content = []
        for fmt in formats:
            if content_data[fmt]:
                rendered_content.append({
                    'title': f"{fmt.upper()}",
                    'rendered': render_content(content_data[fmt], fmt)
                })
                if fmt == "md":
                    html_content = ('<iframe srcdoc="{}" style="width:100%; height:500px; border:none;"></iframe>'.
                                    format(escape(content_data[fmt]).replace('\n', '<br>')))
                    rendered_content.append({
                        'title': f"raw {fmt.upper()}",
                        'rendered': render_content(html_content, "html")
                    })

        html_content = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Example in {}/{}</title>
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
            <h1>Example in {}/{}</h1>
            <p> Each title is a dropdown, you can click on it</p>
        '''.format(domain, version, domain, version)

        for i, item in enumerate(rendered_content):
            html_content += '''
            <div class="dropdown">
                <h2 class="dropdown-header" onclick="toggleContent('content-{}')">{}</h2>
                <div id="content-{}" class="dropdown-content">{}</div>
            </div>
            '''.format(i, escape(item['title']), i, item['rendered'])

        if int(idx) > 0:
            html_content += f'<a href="/v2/content/{domain}/{version}/0">First</a>'
            html_content += ' | '
            html_content += f'<a href="/v2/content/{domain}/{version}/{int(idx) - 1}">Previous</a>'
            html_content += ' | '
        html_content += f'<a href="/v2/content/{domain}/{version}/{int(idx) + 1}">Next</a>'

        html_content += '''
        </body>
        </html>
        '''

        return render_template_string(html_content)
    except Exception as e:
        stack = traceback.format_exc().replace("\n", "<br>")
        error_msg = f'''Server internal Error<br>Error: {e}<br>Stack Trace: {stack}'''
        return error_msg, 500


# Route to check for new/modified files and perform sampling
@bp.route('/check-new/<path:domain>/<path:version>', methods=['GET'])
def check_new(domain, version):
    # make sure version is basepath
    if os.path.basename(version) != version:
        return "Invalid version or domain, they cannot be a path", 400
    try:
        from scripts.example_server.V2.check_new import check_and_sample
        return check_and_sample(domain, version)
    except Exception as e:
        stack = traceback.format_exc().replace("\n", "<br>")
        error_msg = f'''Server internal Error<br>Error: {e}<br>Stack Trace: {stack}'''
        return error_msg, 500
