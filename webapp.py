"""Compatibility entrypoint that forwards to website/webapp.py."""

from website.webapp import app


if __name__ == "__main__":
    app.run(debug=True)
