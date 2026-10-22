"""Execute the notebook and smoke-test the shared Streamlit prediction flow."""
import asyncio
import os
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from module2 import ROOT

def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    path = ROOT/'module2.ipynb'
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    client.create_kernel_manager().kernel_spec.argv[0] = sys.executable
    with tempfile.TemporaryDirectory() as profile:
        client.execute(env={**os.environ, 'IPYTHONDIR': profile})
    nbformat.write(notebook, path)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
    assert not app.exception, app.exception
    app.radio(key='page').set_value('Dự đoán tương tác').run()
    app.button(key='predict').click().run()
    assert not app.exception, app.exception
    assert app.metric[0].value in ['Thấp', 'Trung bình', 'Cao']
    print('Notebook execution and Streamlit prediction passed.')

if __name__ == '__main__':
    main()
