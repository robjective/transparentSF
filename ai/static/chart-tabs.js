// Chart Tabs JavaScript Functionality
function switchChartTab(chartId, tabType) {
    // Hide all panels for this chart
    const localPanel = document.getElementById(chartId + '_local');
    const dwPanel = document.getElementById(chartId + '_dw');
    const localBtn = document.querySelector(`#${chartId}_container .chart-tab-btn[onclick*="'local'"]`);
    const dwBtn = document.querySelector(`#${chartId}_container .chart-tab-btn[onclick*="'dw'"]`);
    
    if (localPanel) localPanel.classList.remove('active');
    if (dwPanel) dwPanel.classList.remove('active');
    if (localBtn) localBtn.classList.remove('active');
    if (dwBtn) dwBtn.classList.remove('active');
    
    // Show the selected panel
    if (tabType === 'local' && localPanel) {
        localPanel.classList.add('active');
        if (localBtn) localBtn.classList.add('active');
    } else if (tabType === 'dw' && dwPanel) {
        dwPanel.classList.add('active');
        if (dwBtn) dwBtn.classList.add('active');
    }
}



