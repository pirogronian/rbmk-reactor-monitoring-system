import redis
import os
from dotenv import load_dotenv

load_dotenv()

# Docker will send "redis" values from .env file
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(
    host=redis_host,
    port=redis_port,
    decode_responses=True
)

# r = redis.Redis(
#     host="localhost",
#     port=6379,
#     decode_responses=True
# )