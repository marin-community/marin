from flask import Flask, render_template_string


def create_app():
    app = Flask(__name__)

    # Import and register blueprints
    from V1.server import bp as bp1
    from V2.server import bp as bp2

    app.register_blueprint(bp1, url_prefix='/v1')
    app.register_blueprint(bp2, url_prefix='/v2')

    @app.route('/')
    def index():
        return render_template_string('''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Example Server</title>
          </head>
          <body>
            <h1>Welcome to the Example Server</h1>
            <li><a href="/v1">V1</a><br>
            <li><a href="/v2">V2</a><br>
          </body>
        </html>
        ''')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
