# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:30:14 2020

@author: Karthik
"""
import os
import hashlib


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)

def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)
    
def download_url(url, root, filename = None, md5 = None):
    """
    Parameters
    ----------
    url : url to the target file.
    root : Root directory to save the file in.
    filename : optional filename with which the file should be saved.
    md5 : md5 checksum of the downloaded file.

    Returns
    -------
    None.
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            filename, _ = urllib.request.urlretrieve(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                filename, _ = urllib.request.urlretrieve(url, fpath)
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")
    return filename

def extract_file(name, to_dir = None):
    """

    Parameters
    ----------
    name : Path to the compressed file.
    to_dir : Folder to which the file is extracted.

    Returns
    -------
    to_dir : Folder in which the file is extracted.

    """
    import zipfile
    import tarfile

    if to_dir is None:
        to_dir = os.path.dirname(name)
    
    if ".tar" in name:
        with tarfile.open(name, 'r') as tar:
            tar.extractall(path = to_dir)
        tar.close()
    if ".zip" in name:
        with zipfile.open(name, 'r') as zipf:
            zipf.extractall(path = to_dir)
        zipf.close()
    print("tar file extracted under dir : {}".format(to_dir))
    return to_dir