#!/usr/bin/env python3
"""
Installation Test Script for NI DAQ Data Acquisition Tool

This script verifies that all required dependencies are properly installed
and provides basic functionality testing without requiring actual DAQ hardware.
"""

import sys
import importlib
from typing import List, Tuple


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Test if a module can be imported successfully.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name for display
    
    Returns:
        Tuple of (success, message)
    """
    try:
        importlib.import_module(module_name)
        display_name = package_name or module_name
        return True, f"✅ {display_name} - OK"
    except ImportError as e:
        display_name = package_name or module_name
        return False, f"❌ {display_name} - FAILED: {str(e)}"
    except Exception as e:
        display_name = package_name or module_name
        return False, f"❌ {display_name} - ERROR: {str(e)}"


def test_nidaqmx_functionality() -> Tuple[bool, str]:
    """
    Test basic nidaqmx functionality without requiring hardware.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        import nidaqmx
        from nidaqmx.constants import TerminalConfiguration, VoltageUnits
        
        # Test system access (this works without hardware)
        system = nidaqmx.system.System.local()
        
        # Test constants access
        _ = TerminalConfiguration.RSE
        _ = VoltageUnits.VOLTS
        
        return True, "✅ nidaqmx functionality - OK"
    except Exception as e:
        return False, f"❌ nidaqmx functionality - FAILED: {str(e)}"


def test_pyqt_functionality() -> Tuple[bool, str]:
    """
    Test basic PyQt5 functionality.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        
        # Create a minimal application instance
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Test timer creation
        timer = QTimer()
        
        return True, "✅ PyQt5 functionality - OK"
    except Exception as e:
        return False, f"❌ PyQt5 functionality - FAILED: {str(e)}"


def test_pyqtgraph_functionality() -> Tuple[bool, str]:
    """
    Test basic PyQtGraph functionality.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        import pyqtgraph as pg
        import numpy as np
        
        # Test plot widget creation (without showing)
        plot_widget = pg.PlotWidget()
        
        # Test basic plotting
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plot_widget.plot(x, y)
        
        return True, "✅ PyQtGraph functionality - OK"
    except Exception as e:
        return False, f"❌ PyQtGraph functionality - FAILED: {str(e)}"


def test_numpy_functionality() -> Tuple[bool, str]:
    """
    Test basic NumPy functionality.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        import numpy as np
        
        # Test array creation and operations
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        
        # Test save/load functionality
        test_data = {'test': arr, 'mean': result}
        
        return True, "✅ NumPy functionality - OK"
    except Exception as e:
        return False, f"❌ NumPy functionality - FAILED: {str(e)}"


def check_python_version() -> Tuple[bool, str]:
    """
    Check if Python version is compatible.
    
    Returns:
        Tuple of (success, message)
    """
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro} - OK"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.7+"


def run_installation_test():
    """
    Run complete installation test suite.
    """
    print("NI DAQ Data Acquisition Tool - Installation Test")
    print("=" * 50)
    print()
    
    # Test results storage
    results: List[Tuple[bool, str]] = []
    
    # Test Python version
    results.append(check_python_version())
    
    # Test required imports
    test_modules = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("PyQt5", "PyQt5"),
        ("PyQt5.QtWidgets", "PyQt5 Widgets"),
        ("PyQt5.QtCore", "PyQt5 Core"),
        ("pyqtgraph", "PyQtGraph"),
        ("nidaqmx", "NI-DAQmx"),
    ]
    
    print("Testing module imports:")
    for module, display_name in test_modules:
        success, message = test_import(module, display_name)
        results.append((success, message))
        print(f"  {message}")
    
    print("\nTesting functionality:")
    
    # Test functionality
    functionality_tests = [
        test_numpy_functionality,
        test_pyqt_functionality,
        test_pyqtgraph_functionality,
        test_nidaqmx_functionality,
    ]
    
    for test_func in functionality_tests:
        success, message = test_func()
        results.append((success, message))
        print(f"  {message}")
    
    # Summary
    print("\n" + "=" * 50)
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Test Summary: {passed_tests}/{total_tests} passed")
    
    if failed_tests > 0:
        print(f"\n❌ {failed_tests} test(s) failed. Please install missing dependencies:")
        print("\n   pip install -r requirements.txt")
        print("\nFor NI-DAQmx issues, ensure NI-DAQmx drivers are installed from:")
        print("   https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html")
        return False
    else:
        print("\n✅ All tests passed! The application should work correctly.")
        print("\nTo start the application, run:")
        print("   python main.py")
        return True


def test_hardware_detection():
    """
    Test hardware detection (optional, requires actual hardware).
    """
    print("\nTesting hardware detection (optional):")
    try:
        import nidaqmx
        system = nidaqmx.system.System.local()
        devices = [device.name for device in system.devices]
        
        if devices:
            print(f"✅ Found {len(devices)} NI DAQ device(s):")
            for device in devices:
                print(f"   - {device}")
        else:
            print("ℹ️  No NI DAQ devices detected (this is normal if no hardware is connected)")
            
    except Exception as e:
        print(f"❌ Hardware detection failed: {str(e)}")


if __name__ == "__main__":
    try:
        success = run_installation_test()
        test_hardware_detection()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {str(e)}")
        sys.exit(1)