"""Run flask App Locally in Debug Mode"""

from lie_to_me import app, socketio


if __name__ == "__main__":
    socketio.run(app, debug=True)
