import app, { db } from "../../firebase";
import { useState } from "react";
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { addDoc, collection } from "firebase/firestore";

// Find all the prefixes and items.
export function ListOfFiles() {
  const storage = getStorage(app);
  const [file, setFile] = useState("");

  const upload = () => {
    if (file == null) return;
    // Sending File to Firebase Storage
    const storageRef = ref(storage, `/sales_data_csv_files/${file.name}`);

    // 'file' comes from the Blob or File API
    uploadBytes(storageRef, file).then(
      (snapshot) => {
        console.log("Uploaded a blob or file!");
        alert("File Uploaded Successfully");
        // Handle successful uploads on complete
        // For instance, get the download URL: https://firebasestorage.googleapis.com/...
        getDownloadURL(snapshot.ref).then((downloadURL) => {
          // console.log("File available at", downloadURL);
          // Sending File to Firebase Firestore
          addDoc(collection(db, "sales_data_csv_files"), {
            file_name: file.name,
            file_size: file.size,
            file_type: file.type,
            file_url: downloadURL,
          });
        });
      },
      (error) => {
        // Handle unsuccessful uploads
        console.log(error);
      }
    );
  };
  return (
    <div className="ListAll" style={{ marginTop: 250 }}>
      <center>
        <input
          type="file"
          className="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded inline-flex items-center w-80"
          onChange={(e) => {
            setFile(e.target.files[0]);
          }}
        />
        <button
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ml-2"
          onClick={upload}
        >
          Upload
        </button>
        <br />
      </center>
    </div>
  );
}
