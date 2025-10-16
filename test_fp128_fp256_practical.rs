use safetensors::tensor::{serialize, SafeTensors, Dtype, TensorView};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing FP128 and FP256 practical usage...");
    
    // Test FP128 with actual data
    println!("\n=== Testing FP128 ===");
    let fp128_data: Vec<u8> = vec![0u8; 32]; // 2 elements * 16 bytes each
    let fp128_shape = vec![2];
    
    match TensorView::new(Dtype::F128, fp128_shape.clone(), &fp128_data) {
        Ok(tensor) => {
            println!("✓ FP128 TensorView created successfully");
            println!("  Shape: {:?}", tensor.shape());
            println!("  Dtype: {:?}", tensor.dtype());
            println!("  Data size: {} bytes", tensor.data().len());
            
            // Test serialization
            let mut tensors: HashMap<String, TensorView> = HashMap::new();
            tensors.insert("fp128_test".to_string(), tensor);
            
            match serialize(&tensors, None) {
                Ok(serialized) => {
                    println!("✓ FP128 serialization successful, size: {} bytes", serialized.len());
                    
                    // Test deserialization
                    match SafeTensors::deserialize(&serialized) {
                        Ok(deserialized) => {
                            println!("✓ FP128 deserialization successful");
                            let tensor = deserialized.tensor("fp128_test").unwrap();
                            println!("  Deserialized shape: {:?}", tensor.shape());
                            println!("  Deserialized dtype: {:?}", tensor.dtype());
                        }
                        Err(e) => println!("✗ FP128 deserialization failed: {:?}", e),
                    }
                }
                Err(e) => println!("✗ FP128 serialization failed: {:?}", e),
            }
        }
        Err(e) => println!("✗ FP128 TensorView creation failed: {:?}", e),
    }
    
    // Test FP256 with actual data
    println!("\n=== Testing FP256 ===");
    let fp256_data: Vec<u8> = vec![0u8; 64]; // 2 elements * 32 bytes each
    let fp256_shape = vec![2];
    
    match TensorView::new(Dtype::F256, fp256_shape.clone(), &fp256_data) {
        Ok(tensor) => {
            println!("✓ FP256 TensorView created successfully");
            println!("  Shape: {:?}", tensor.shape());
            println!("  Dtype: {:?}", tensor.dtype());
            println!("  Data size: {} bytes", tensor.data().len());
            
            // Test serialization
            let mut tensors: HashMap<String, TensorView> = HashMap::new();
            tensors.insert("fp256_test".to_string(), tensor);
            
            match serialize(&tensors, None) {
                Ok(serialized) => {
                    println!("✓ FP256 serialization successful, size: {} bytes", serialized.len());
                    
                    // Test deserialization
                    match SafeTensors::deserialize(&serialized) {
                        Ok(deserialized) => {
                            println!("✓ FP256 deserialization successful");
                            let tensor = deserialized.tensor("fp256_test").unwrap();
                            println!("  Deserialized shape: {:?}", tensor.shape());
                            println!("  Deserialized dtype: {:?}", tensor.dtype());
                        }
                        Err(e) => println!("✗ FP256 deserialization failed: {:?}", e),
                    }
                }
                Err(e) => println!("✗ FP256 serialization failed: {:?}", e),
            }
        }
        Err(e) => println!("✗ FP256 TensorView creation failed: {:?}", e),
    }
    
    Ok(())
}
