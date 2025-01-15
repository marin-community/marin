import server as server_module
import yaml
from server import app, ServerConfig, standardize_config, Server

# Read config from gcp.conf
with open('conf/gcp.conf', 'r') as f:
    config_dict = yaml.safe_load(f)

# Initialize server configuration
config = ServerConfig(**config_dict)
config = standardize_config(config)

# Initialize global server instance
server_module.server = Server(config)

# This is what Gunicorn will import
application = app 