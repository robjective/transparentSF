// Reset theme to light mode
// Run this in the browser console to reset the theme

(function() {
    'use strict';
    
    // Clear localStorage theme
    localStorage.removeItem('theme');
    
    // Set theme to light
    localStorage.setItem('theme', 'light');
    
    // Update the page if dark mode manager exists
    if (window.darkModeManager) {
        window.darkModeManager.resetToLightMode();
    } else {
        // Set the theme attribute directly
        document.documentElement.setAttribute('data-theme', 'light');
    }
    
    console.log('Theme reset to light mode');
    
    // Reload the page to ensure clean state
    setTimeout(() => {
        window.location.reload();
    }, 500);
    
})(); 