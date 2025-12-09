import React from 'react';
import logoImage from './assets/images/logo.png'; // Đường dẫn đến ảnh logo của bạn
//import './Header.css'; // File CSS riêng

function Header() {
  return (
    <header className="header">
      <div className="logo-container">
        <img 
          src={logoImage} 
          alt="Arbin Instruments Logo" 
          className="logo"
          style={{ width: '100px', height: 'auto' }}
        />
        {/* Giữ text logo nếu muốn */}
        {/*<span className="logo-text">Arbin Chatbot</span>*/}
      </div>
      <div className="header-info">
        <h1>Arbin Instruments Assistant</h1>
        <p>Ask questions about Arbin products and documentation</p>
      </div>
    </header>
  );
}

export default Header;