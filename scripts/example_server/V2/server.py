import json
import os
import traceback

import fsspec
import grip
from flask import Flask, render_template_string
from markupsafe import escape

app = Flask(__name__)
githubRenderer = grip.GitHubRenderer()


# Function to render content based on type
def render_content(content, format_type):
    if format_type == 'text' or format_type == 'txt':
        return '<p>{}</p>'.format(escape(content))
    elif format_type == 'html' or format_type == 'xml':
        iframe_content = f'<iframe srcdoc="{escape(content)}" style="width:100%; height:500px; border:none;"></iframe>'
        return iframe_content
    elif format_type == 'md':
        return githubRenderer.render(content)
    else:
        return '<p>Unsupported content type, Rending as plain text <br> {}</p>'.format(escape(content))


# Route to display the home page
@app.route('/', methods=['GET'])
def display_home():
    fs = fsspec.filesystem("gcs")
    files = fs.ls("gcs://marin-data/examples/", use_listings_cache=False, detail=True)
    # import pdb; pdb.set_trace()
    domains = [os.path.basename(f["name"]) for f in files if f["type"] == "directory"]
    versions = {}
    for domain in domains:
        versions[domain] = [os.path.basename(f) for f in
                            fs.ls(f"gcs://marin-data/examples/{domain}", use_listings_cache=False)]

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
            html_content += f'''<li><a href="/content/{domain}/{version}/0">{domain}, {version}</a><br>'''
    html_content += '''
    </body>
    </html>
    '''
    return render_template_string(html_content)


# Route to display the content
@app.route('/content/<path:domain>/<path:version>/<int:idx>', methods=['GET'])
def display_content(domain, version, idx):
    try:
        content_data = {}
        fs = fsspec.filesystem("gcs")
        base_path = f"gcs://marin-data/examples/{domain}/{version}/"
        formats = fs.ls(base_path, use_listings_cache=False)
        formats = [fmt for fmt in formats if not os.path.basename(fmt).lower().endswith('.json')]
        formats = [os.path.basename(fmt) for fmt in formats if
                   os.path.basename(fmt).lower().startswith(('xml', 'html', 'md', 'txt', "text"))]
        # formats = [fmt.split('/')[-2] for fmt in formats]

        for fmt in formats:
            filename = f"gcs://marin-data/examples/{domain}/{version}/{fmt}/samples.jsonl"
            i = 0
            with fsspec.open(filename, 'rt', use_listings_cache=False) as file:
                for line in file:
                    if i == idx:
                        record = json.loads(line.strip())
                        content_data[fmt] = record['text']
                        break
                    i += 1
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
            html_content += f'<a href="/content/{domain}/{version}/0">First</a>'
            html_content += ' | '
            html_content += f'<a href="/content/{domain}/{version}/{int(idx) - 1}">Previous</a>'
            html_content += ' | '
        html_content += f'<a href="/content/{domain}/{version}/{int(idx) + 1}">Next</a>'

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
@app.route('/check-new/<path:domain>/<path:version>', methods=['GET'])
def check_new(domain, version):
    # make sure version is basepath
    if os.path.basename(version) != version:
        return "Invalid version or domain, they cannot be a path", 400
    try:
        from check_new import check_and_sample
        return check_and_sample(domain, version)
    except Exception as e:
        stack = traceback.format_exc().replace("\n", "<br>")
        error_msg = f'''Server internal Error<br>Error: {e}<br>Stack Trace: {stack}'''
        return error_msg, 500


if __name__ == '__main__':
    # Check if the deployment variable is set (to run on the server)
    if 'DEPLOYMENT' in os.environ:
        app.run(host='0.0.0.0', port=5000)
    else:
        app.run(debug=True)
