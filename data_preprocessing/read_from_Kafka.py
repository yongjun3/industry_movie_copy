from kafka import KafkaConsumer
import json
import os
import logging
from data_quality import data_quality_monitor

# Configure logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='log/kafka_reader.log'
)
logger = logging.getLogger('kafka_reader')

# Kafka connection settings
topic = 'movielog27'
output_file = "data/output.txt"
valid_output_file = "data/valid_output.txt"
invalid_output_file = "data/invalid_output.txt"

# Create directory for data storage
os.makedirs("data", exist_ok=True)

try:
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=["localhost:9092"],
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda x: x.decode("utf-8"),
        consumer_timeout_ms=10000
    )
    
    logger.info(f"Connected to Kafka topic: {topic}, streaming messages to file...")
    print(f"Connected to Kafka topic: {topic}, streaming messages to file...")

    # Read messages from Kafka with validation
    with open(output_file, "a") as file, \
         open(valid_output_file, "a") as valid_file, \
         open(invalid_output_file, "a") as invalid_file:
        
        message_count = 0
        valid_count = 0
        invalid_count = 0
        
        for message in consumer:
            # Process the message
            log_entry = message.value
            message_count += 1
            
            # Write raw output
            file.write(log_entry + "\n")
            
            # Validate the log entry using data_quality_monitor
            is_valid = data_quality_monitor.validate_log_entry(log_entry)
            
            if is_valid:
                valid_count += 1
                valid_file.write(log_entry + "\n")
                logger.info(f"Valid log entry: {log_entry}")
            else:
                invalid_count += 1
                invalid_file.write(log_entry + "\n")
                logger.warning(f"Invalid log entry: {log_entry}")
            
            # Print progress periodically
            if message_count % 10000 == 0:
                print(f"Processed {message_count} messages: {valid_count} valid, {invalid_count} has invalid schema.")
                
                # Log schema quality metrics
                schema_metrics = data_quality_monitor.get_schema_quality_metrics()
                logger.info(f"Schema quality metrics: {json.dumps(schema_metrics)}")
                
                # Calculate violation rate
                violation_rate = schema_metrics['violation_rate'] * 100
                print(f"Current violation rate: {violation_rate:.2f}%")

    # Final reporting
    print("Finished reading all available messages.")
    print(f"Total messages processed: {message_count}")
    print(f"Valid messages: {valid_count} ({valid_count/message_count*100:.2f}%)")
    print(f"Invalid messages: {invalid_count} ({invalid_count/message_count*100:.2f}%)")
    
    # Log final schema quality metrics
    schema_metrics = data_quality_monitor.get_schema_quality_metrics()
    logger.info(f"Final schema quality metrics: {json.dumps(schema_metrics)}")

except Exception as e:
    logger.error(f"Error in Kafka consumer: {str(e)}")
    print(f"Error: {str(e)}")