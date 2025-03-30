import React from "react";
import { Routes, Route } from "react-router-dom";
import Login from "./components/navbar/login";
import Register from "./components/navbar/registation";

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
    </Routes>
  );
}

export default App;
