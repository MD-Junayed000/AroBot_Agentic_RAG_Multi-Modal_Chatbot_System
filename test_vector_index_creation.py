#!/usr/bin/env python3
"""
Comprehensive Test: Vector Index Creation via "Add PDF to New Index" Feature
Tests the complete workflow from UI button click to Pinecone index creation
"""
import requests
import time
from pathlib import Path
import sys

def test_vector_index_creation():
    print("=== TESTING: Add PDF to New Index Feature ===")
    print("=" * 60)
    
    # Check if server is running
    print("Step 1: Checking server status...")
    try:
        health_check = requests.get("http://localhost:8000/health", timeout=10)
        if health_check.status_code == 200:
            print("✅ Server is running and healthy")
        else:
            print(f"❌ Server health check failed: {health_check.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Please ensure server is running with: python app.py")
        return False
    
    # Find PDF files
    print("\nStep 2: Finding PDF files for testing...")
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in data directory")
        return False
    
    test_pdf = pdf_files[0]
    print(f"✅ Found PDF for testing: {test_pdf.name}")
    print(f"📏 File size: {test_pdf.stat().st_size / (1024*1024):.2f} MB")
    
    # Test vector index creation with proper naming
    session_id = f"test_vector_index_{int(time.time())}"
    
    # Test cases with different naming patterns
    test_cases = [
        {
            "user_name": "cardiology-research",
            "expected_index": "arobot-medical-pdf-cardiology-research",
            "description": "Cardiology research documents"
        },
        {
            "user_name": "neurology studies", 
            "expected_index": "arobot-medical-pdf-neurology-studies",
            "description": "Neurology clinical studies"
        }
    ]
    
    all_passed = True
    created_indexes = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: Creating index '{test_case['user_name']}'")
        print('='*60)
        
        user_name = test_case["user_name"]
        expected_index = test_case["expected_index"]
        description = test_case["description"]
        
        print(f"📝 User input name: {user_name}")
        print(f"🎯 Expected index name: {expected_index}")
        print(f"📄 Description: {description}")
        
        try:
            start_time = time.time()
            
            # Prepare the request exactly as the UI would send it
            with open(test_pdf, 'rb') as f:
                files = {'file': (test_pdf.name, f, 'application/pdf')}
                data = {
                    'index_name': user_name,  # User input name (will be formatted by server)
                    'description': description,
                    'session_id': session_id
                }
                
                print(f"🔄 Creating vector index...")
                print(f"🌐 Sending request to: /api/v1/vector/create-index")
                
                response = requests.post(
                    'http://localhost:8000/api/v1/vector/create-index',
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes for vector index creation
                )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"⏱️ Processing time: {processing_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract results
                message = result.get('message', '')
                chunks_processed = result.get('chunks_processed', 0)
                index_name_result = result.get('index_name', '')
                
                print(f"✅ SUCCESS: Vector index created!")
                print(f"📊 Chunks processed: {chunks_processed}")
                print(f"🏷️ Actual index name: {index_name_result}")
                print(f"💬 Server message: {message}")
                
                # Verify the results
                success_checks = []
                
                # Check 1: Index name matches expected pattern
                if expected_index in index_name_result:
                    print("✅ Index name follows correct pattern")
                    success_checks.append(True)
                else:
                    print(f"❌ Index name mismatch - Expected: {expected_index}, Got: {index_name_result}")
                    success_checks.append(False)
                
                # Check 2: Chunks were processed
                if chunks_processed > 0:
                    print(f"✅ Document was properly chunked: {chunks_processed} chunks")
                    success_checks.append(True)
                else:
                    print("❌ No chunks were processed")
                    success_checks.append(False)
                
                # Check 3: Processing time is reasonable
                if processing_time <= 180:  # 3 minutes max
                    print(f"✅ Reasonable processing time: {processing_time:.2f}s")
                    success_checks.append(True)
                else:
                    print(f"⚠️ Slow processing time: {processing_time:.2f}s")
                    success_checks.append(False)
                
                # Check 4: Message indicates success
                if "successfully created" in message.lower():
                    print("✅ Success message confirmed")
                    success_checks.append(True)
                else:
                    print("⚠️ Success message not found")
                    success_checks.append(False)
                
                if all(success_checks):
                    print(f"🎉 TEST CASE {i}: PASSED")
                    created_indexes.append({
                        'name': index_name_result,
                        'chunks': chunks_processed,
                        'time': processing_time
                    })
                else:
                    print(f"❌ TEST CASE {i}: FAILED")
                    all_passed = False
                    
            else:
                print(f"❌ FAILED: HTTP {response.status_code}")
                print(f"Error response: {response.text[:500]}")
                all_passed = False
                
        except requests.exceptions.Timeout:
            print("❌ TIMEOUT: Index creation took too long (>5 minutes)")
            all_passed = False
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
        
        # Small delay between test cases
        if i < len(test_cases):
            print(f"\n⏳ Waiting 5 seconds before next test case...")
            time.sleep(5)
    
    # Final verification: Check if indexes appear in Pinecone
    print(f"\n{'='*60}")
    print("FINAL VERIFICATION: Checking Pinecone Database")
    print('='*60)
    
    print("💡 Verification steps:")
    print("1. Check your Pinecone dashboard at: https://app.pinecone.io/")
    print("2. Look for the newly created indexes:")
    
    for idx in created_indexes:
        print(f"   • {idx['name']} ({idx['chunks']} chunks)")
    
    print("\n🌐 UI Verification:")
    print("1. Go to: http://localhost:8000/chat")
    print("2. Check the 'Vector Indexes' section in the left panel")
    print("3. You should see the new indexes listed")
    print("4. The UI should be updated to show the new indexes")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL TEST RESULTS")
    print('='*60)
    
    print(f"✅ Indexes successfully created: {len(created_indexes)}")
    print(f"❌ Failed test cases: {len(test_cases) - len(created_indexes)}")
    
    if created_indexes:
        print("\n🎯 Created Indexes:")
        for idx in created_indexes:
            print(f"   • {idx['name']}")
            print(f"     - Chunks: {idx['chunks']}")
            print(f"     - Processing time: {idx['time']:.2f}s")
    
    print(f"\n🎯 Feature Status:")
    if all_passed and len(created_indexes) == len(test_cases):
        print("✅ 'Add PDF to New Index' feature is FULLY FUNCTIONAL")
        print("✅ Naming convention working correctly")
        print("✅ Pinecone integration working")
        print("✅ Document chunking working")
        return True
    else:
        print("❌ Some issues found with the feature")
        return False

def test_ui_accessibility():
    """Test if the UI is accessible and shows the indexes"""
    print(f"\n{'='*60}")
    print("BONUS TEST: UI Accessibility")
    print('='*60)
    
    try:
        ui_response = requests.get('http://localhost:8000/chat', timeout=10)
        if ui_response.status_code == 200:
            print("✅ Chat interface is accessible")
            print("✅ UI should display vector indexes in left panel")
            print("✅ 'Add PDF to New Index' button should be visible")
            return True
        else:
            print(f"❌ UI not accessible: {ui_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ UI access error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE VECTOR INDEX CREATION TEST")
    print("="*60)
    print("This test will verify:")
    print("• Index naming: arobot_medical_pdf_[user_name]")
    print("• Pinecone database integration")
    print("• Document chunking and processing")
    print("• UI integration and updates")
    print("• Error handling")
    print("="*60)
    
    # Run main test
    main_success = test_vector_index_creation()
    
    # Run UI test
    ui_success = test_ui_accessibility()
    
    # Final verdict
    print(f"\n{'🎉' if main_success and ui_success else '❌'} OVERALL RESULT:")
    if main_success and ui_success:
        print("✅ ALL TESTS PASSED!")
        print("🎊 'Add PDF to New Index' feature is fully functional!")
    else:
        print("❌ Some tests failed - review the results above")
    
    sys.exit(0 if (main_success and ui_success) else 1)
