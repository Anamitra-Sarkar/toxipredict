import { initializeApp, getApps } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDU4EEHT3HEvKNPOrpglLdF3y5Tfs6qy4E",
  authDomain: "plant-cloud-cd461.firebaseapp.com",
  projectId: "plant-cloud-cd461",
  storageBucket: "plant-cloud-cd461.firebasestorage.app",
  messagingSenderId: "130972851055",
  appId: "1:130972851055:web:9a2d631c1736eb80dca69a",
  measurementId: "G-38VMJ6EXMS",
};

const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
