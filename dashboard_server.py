import http.server
import socketserver

# Define the port you want to use
PORT = 5555

# Set up the request handler to serve files from the current directory
Handler = http.server.SimpleHTTPRequestHandler

# Initialize the server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()
