import os
import sys
import traceback

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore, storage


def initialize_firestore():
    load_dotenv()
    cred_path = os.getenv('FIREBASE_PATH')
    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    path_to_dat = os.path.abspath(os.path.join(bundle_dir, cred_path))
    # print("divi test 2", path_to_dat)
    cred = credentials.Certificate("path/to/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)


def get_firestore_files(search_file_name):
    try:
        firestore.client()
    except Exception as e:
        print(e)
        initialize_firestore()
    db = firestore.client()
    docs = db.collection(u'sales_data_csv_files').where(u'file_name', u'>=', search_file_name). \
        where("file_name", "<=", search_file_name + "\uf8ff"). \
        limit(10). \
        stream()
    return docs


def upload_file_to_firestore(file):
    try:
        storage.bucket("abhicapstoneproject.appspot.com")
    except Exception as e:
        print(e)
        initialize_firestore()
    try:
        bucket = storage.bucket("abhicapstoneproject.appspot.com")
        blob = bucket.blob('sales_data_csv_files/'+file.name)
        blob.upload_from_file(file)
        blob.make_public()
        file_url = blob.public_url
        db = firestore.client()
        doc_ref = db.collection(u'sales_data_csv_files').document()
        doc_ref.set({
            u'file_name': file.name,
            u'file_url': file_url,
            u'file_type': file.type,
            u'file_size': file.size
        })
        return file_url
    except Exception as e:
        print(e)
        traceback.print_exc()
        return None
