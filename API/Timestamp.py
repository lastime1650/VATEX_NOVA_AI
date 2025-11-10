import datetime, time

def now_to_nano_int() -> int:
    return time.time_ns()

def nano_to_iso_string(nanoseconds: int) -> str:
    """
    19자리 나노초 정수를 ISO 8601 UTC 문자열로 변환합니다.
    예: 1730105630112233500 -> "2024-10-28T09:33:50.112233500Z"
    """
    # 초(seconds)와 나노초(nanoseconds part)로 분리
    seconds = nanoseconds // 1_000_000_000
    nanos_part = nanoseconds % 1_000_000_000
    
    # 이 시스템이 존재한 특정 나라 UTC 기준으로 datetime 객체 생성
    dt_object = datetime.datetime.fromtimestamp(seconds).astimezone()
    
    # YYYY-MM-DDTHH:MM:SS 형식으로 포맷팅
    base_format = dt_object.strftime('%Y-%m-%dT%H:%M:%S')
    
    # 나노초 부분을 9자리로 맞춰서 결합하고 Z를 붙여 UTC임을 명시
    return f"{base_format}.{nanos_part:09d}Z"