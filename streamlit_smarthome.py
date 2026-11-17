"""로컬 스마트홈 Streamlit 진입점."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st


API_BASE_URL = os.environ.get("SMART_HOME_API_URL", "http://127.0.0.1:8000").rstrip("/")
# 서버의 공유 135초 채팅 마감보다 긴 HTTP 대기 한도다. 모델은 서버 시작 때 준비한다.
POST_TIMEOUT_SECONDS = 360
CHAT_STATUS = {
    "ok": "성공", "noop": "변경 없음", "proposed": "제안",
    "clarify": "확인 필요", "blocked": "차단", "unavailable": "사용 불가",
}
SERVICE_STATUS = {"ready": "준비", "not_loaded": "로드 전", "unavailable": "사용 불가"}
HAR_STATUS = {"predicted": "추정 완료", "abstained": "판단 보류", "unavailable": "사용 불가"}
T06_DECISION = {"analyze": "추가 분석", "log_only": "기록만", "abstain": "판단 보류"}


class ApiError(Exception):
    def __init__(self, status, message, body=None):
        super().__init__(message)
        self.status = status
        self.body = body if isinstance(body, dict) else {}
        self.trace = self.body.get("trace", [])


def api_request(path, *, user_id=None, payload=None):
    url = API_BASE_URL + path
    if user_id is not None:
        url += "?" + urlencode({"user_id": user_id})
    request = Request(url, headers={"Accept": "application/json"})
    if payload is not None:
        request.data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=POST_TIMEOUT_SECONDS if payload is not None else 10) as response:
            return json.load(response)
    except HTTPError as error:
        messages = {
            409: "이미 사용한 전송 ID의 메시지와 일치하지 않습니다.",
            422: "요청 값을 확인해 주세요.",
            503: "데이터베이스에서 요청을 처리하거나 확정할 수 없습니다.",
        }
        try:
            body = json.load(error)
        except (ValueError, OSError):
            body = {}
        message = body.get("message") or body.get("detail") if isinstance(body, dict) else None
        if not isinstance(message, str):
            message = messages.get(error.code, "서버가 요청을 처리하지 못했습니다.")
        raise ApiError(error.code, message, body) from None
    except (URLError, OSError, ValueError):
        raise ApiError(None, "로컬 API의 응답을 확인하지 못했습니다. 실행 여부는 확인 불가이며 같은 요청 ID로 결과를 확인하세요.") from None


def send_chat(profile):
    pending = profile["pending"]
    try:
        result = api_request("/chat", payload=pending)
    except ApiError as error:
        result = {"status": "unavailable", "code": "transport_unconfirmed",
                  "message": str(error), **error.body, "trace": error.trace}
        profile["messages"].append(("assistant", result))
        if error.status in (409, 422):
            profile["pending"] = None
        else:
            profile["retry_reason"] = str(error)
        return
    profile["messages"].append(("assistant", result))
    profile["pending"] = None
    profile["retry_reason"] = None


def render_trace(trace):
    with st.expander("처리 과정"):
        if trace:
            st.table([{"구성 요소": item["component"], "상태": item["status"], "요약": item["summary"]}
                      for item in trace])
        else:
            st.caption("서버에서 제공한 처리 과정이 없습니다.")


def render_selected_memory(personalization):
    selected = (personalization or {}).get("selected")
    if not selected:
        return
    st.caption("선택한 설정 출처 · 검색된 상위 12건 안에서 최근 적합 기록 우선")
    st.json({"근거 ID": selected["memory_ids"], "기기": selected["device_id"],
             "settings": selected["settings"], "원천": selected["source_kind"],
             "원문": selected["memory_text"], "선택 근거 시각": selected["latest_at"]})


def render_memory_write(memory_write):
    if not memory_write:
        st.caption("이전 이력 · 기억 저장 결과 필드 없음")
        return
    labels = {"stored": "저장", "skipped": "미저장", "failed": "저장 실패"}
    st.caption("개인 기억 · " + labels.get(memory_write.get("status"), "확인 불가"))
    if memory_write.get("status") == "stored":
        st.caption(f"기억 ID {memory_write['memory_id']} · {memory_write['source_kind']} · {memory_write['source_id']}")
    else:
        st.caption("사유 · " + memory_write.get("reason", "확인 불가"))


def render_result(result, state):
    label = "근거 부족" if result.get("code") == "no_preference_memory" else CHAT_STATUS.get(result.get("status"), "요청 오류")
    st.text(f'{label} · {result["message"]}')
    if result.get("preparation_notice"):
        st.warning(result["preparation_notice"])
    if result.get("status") == "proposed":
        st.caption("과거 제안 이력 · 새로 적용된 결과가 아닙니다.")
    device = result.get("device")
    if device:
        power = "켜짐" if device["power"] else "꺼짐"
        brightness = device.get("brightness_pct")
        st.caption(f'실행 당시 기기 · {device["device_id"]} · {power}'
                   + (f" · 밝기 {brightness}%" if device["power"] and brightness is not None else ""))
    render_selected_memory(result.get("personalization"))
    render_memory_write(result.get("memory_write"))
    render_trace(result.get("trace", []))


def render_chat(user_id, profile, state):
    st.header("대화")
    for role, message in profile["messages"]:
        with st.chat_message(role):
            if role == "assistant" and isinstance(message, dict):
                render_result(message, state)
            else:
                st.text(message)
    if profile["pending"] is not None:
        st.warning(profile["retry_reason"] or "전송 결과를 확인할 수 없습니다.")
        if st.button("같은 메시지 다시 전송"):
            with st.spinner("기존 요청 ID로 결과 확인 중…"):
                send_chat(profile)
            st.rerun()
    message = st.chat_input("가상 기기에 요청하기", disabled=profile["pending"] is not None)
    if message and message.strip():
        profile["messages"].append(("user", message))
        profile["pending"] = {"user_id": user_id, "request_id": str(uuid4()), "message": message}
        with st.spinner("요청 처리 중…"):
            send_chat(profile)
        st.rerun()


def render_house(state):
    st.header("가상 집")
    st.caption("현재 집 상태 · 서버에서 새로 조회한 값입니다.")
    for room in state["rooms"]:
        with st.container(border=True):
            st.subheader(room["label"])
            st.caption("센서 · " + ", ".join(room["sensor_ids"]))
            for device in state["devices"]:
                if device["room_id"] != room["room_id"]:
                    continue
                power = "켜짐" if device["power"] else "꺼짐"
                if device["power"] and device["brightness_pct"] is not None:
                    power += f' · 밝기 {device["brightness_pct"]}%'
                st.text(f'{device["label"]} · {power} ({device["device_id"]})')


def sensor_event_rows(events, rooms):
    sensor_labels = {}
    for room in rooms:
        motion_number = 0
        for sensor_id in room["sensor_ids"]:
            if sensor_id.startswith("M"):
                motion_number += 1
                label = f"움직임 센서 {motion_number}"
            elif sensor_id == "D001":
                label = "현관문 개폐 센서"
            else:
                label = "센서"
            sensor_labels[sensor_id] = (room["label"], label)

    rows = []
    for event in events:
        sensor_id = event["sensor_id"]
        room_label, sensor_label = sensor_labels.get(sensor_id, ("미등록 공간", "미등록 센서"))
        row = {
            "시각(UTC)": event["timestamp"],
            "공간": room_label,
            "센서": sensor_label,
            "센서 ID": sensor_id,
            "상태": event["state"],
        }
        if "source" in event:
            row["출처"] = event["source"]
        rows.append(row)
    return rows


def render_har(latest, rooms):
    st.subheader("집 전체의 최근 활동 추정")
    if latest is None:
        st.info("아직 센서 시연 결과가 없습니다.")
        return
    har, t06 = latest["har"], latest["t06"]
    st.text("기대 활동 · " + (latest["expected_activity"] or "입력 부족 시나리오"))
    st.text("HAR · " + HAR_STATUS[har["status"]])
    if har["activity"] is not None:
        st.text("실제 추정 · " + har["activity"])
    if har["reason"] is not None:
        st.caption("HAR 사유 · " + har["reason"])
    if har["window_start"] is not None and har["window_end"] is not None:
        st.caption(f'관측(UTC) · {har["window_start"]} ~ {har["window_end"]}')
    if har["model_version"] is not None:
        st.caption("모델 버전 · " + har["model_version"])
    st.text("T06 · " + T06_DECISION[t06["decision"]])
    if t06["reason"] is not None:
        st.caption("T06 사유 · " + t06["reason"])
    with st.expander("합성 센서 이벤트"):
        st.table(sensor_event_rows(latest["events"], rooms))


def render_activity(user_id, state, har_service):
    st.subheader("센서·HAR")
    presets = {item["preset_id"]: item["label"] for item in state["har_presets"]}
    preset_id = st.selectbox("센서 시나리오", list(presets), format_func=presets.__getitem__)
    if preset_id in har_service.get("unsupported_activities", []):
        st.warning("현재 HAR 모델의 학습 데이터에서 이 활동을 지원하지 않습니다.")
    if st.button("센서 시연 실행"):
        try:
            api_request("/har/demo", payload={"user_id": user_id, "preset_id": preset_id})
            st.rerun()
        except ApiError as error:
            st.error(str(error))
    render_har(state["latest_har"], state["rooms"])


PERSONALIZATION_SCENARIOS = {
    "휴식 조명": "오늘 좀 쉬고 싶은데 조명을 어떻게 할까?",
    "야간 작업": "밤에 서재에서 노트북 작업하려는데 뭘 준비하면 좋을까?",
}


def send_personalization(profile):
    pending = profile["personalization_pending"]
    try:
        response = api_request("/demo/personalization", payload=pending)
    except ApiError as error:
        result = {"request_id": pending["request_id"], "status": "unavailable",
                  "code": "transport_unconfirmed", "message": str(error),
                  **error.body, "trace": error.trace}
        evidence = result.get("personalization") or {}
        profile["personalization"] = {
            **pending, "embedding_model": evidence.get("embedding_model"),
            "embedding_dimensions": evidence.get("embedding_dimensions"),
            "memories": evidence.get("memories", []), "candidates": evidence.get("candidates", []),
            "selected": evidence.get("selected"), "result": result,
        }
        if error.status in (409, 422):
            profile["personalization_pending"] = None
            profile["personalization_retry_reason"] = None
        else:
            profile["personalization_retry_reason"] = str(error)
        return
    profile["personalization"] = response
    profile["personalization_pending"] = None
    profile["personalization_retry_reason"] = None


def render_personalization(user_id, profile, state):
    st.subheader("사용 기록·개인화")
    profile.setdefault("personalization_pending", None)
    profile.setdefault("personalization_retry_reason", None)
    pending = profile["personalization_pending"]
    scenario = st.selectbox("개인화 시나리오", list(PERSONALIZATION_SCENARIOS), key=f"scenario_{user_id}")
    query = st.text_input("검색 질문", value=PERSONALIZATION_SCENARIOS[scenario], key=f"query_{user_id}_{scenario}")
    st.caption("내 기록에 맞춰 가상 기기 설정을 바로 적용합니다")
    if pending is not None:
        st.warning(profile["personalization_retry_reason"] or "요청 결과 확인 중입니다.")
        st.caption(f'확인할 요청 · 사용자 {pending["user_id"]} · {pending["query"]} · {pending["request_id"]}')
        if st.button("같은 요청 결과 확인", key=f"retry_personalization_{user_id}"):
            with st.spinner("기존 요청 ID로 결과 확인 중…"):
                send_personalization(profile)
            st.rerun()
    if st.button("사용 기록으로 설정 적용", disabled=pending is not None or not query.strip()):
        profile["personalization_pending"] = {
            "user_id": user_id, "request_id": str(uuid4()), "query": query,
        }
        with st.spinner("개인 사용 기록 검색과 설정 적용 중…"):
            send_personalization(profile)
        st.rerun()
    response = profile.get("personalization")
    if response is None:
        st.caption("버튼을 누르면 본인 기록을 검색하고 선택한 설정을 한 번 적용합니다.")
        return
    st.subheader("마지막 실행 결과")
    st.caption("실행 당시 결과입니다. 현재 상태는 가상 집 탭에서 별도로 확인하세요.")
    st.text(f'사용자 {response["user_id"]} · {response["query"]}')
    st.caption(f'요청 ID · {response.get("request_id", "이전 이력") }')
    if response.get("embedding_model") is not None:
        st.caption(f'multilingual-e5-small · {response["embedding_dimensions"]}차원 · pgvector 코사인 검색')
        st.caption("코사인 거리는 낮을수록 비슷한 기록입니다.")
    else:
        st.caption("이 요청에서 임베딩 검색 정보를 확보하지 못했습니다.")
    st.caption("합성 사용 기록과 사용자 요청 기록을 검색합니다. 자동 개인화 실행 기억은 후속 선호 검색에서 제외됩니다.")
    if response.get("memories"):
        rows = sorted(response["memories"], key=lambda row: row["distance"])[:3]
        source_labels = {"dummy_usage": "합성 사용 기록", "user_request": "사용자 요청 기록"}
        st.table([{"기억 ID": row["memory_id"], "원문": row["memory_text"],
                   "기기": row.get("device_id"), "settings": json.dumps(row.get("settings"), ensure_ascii=False),
                   "원천": source_labels.get(row.get("source_kind"), row.get("source_kind")),
                   "시각": row.get("occurred_at"), "코사인 거리": round(row["distance"], 5)} for row in rows])
    else:
        st.info("이 요청에서 확보한 검색 기록이 없습니다. 아래 결과와 준비 상태를 확인하세요.")
    render_result(response["result"], state)


def render_logs(user_id, state):
    st.header("실행 로그")
    st.subheader(f"{user_id}의 활성 유지 작업")
    if state["active_tasks"]:
        st.table(state["active_tasks"])
    else:
        st.caption("활성 유지 작업이 없습니다.")
    st.subheader("공용 기기 변경")
    if state["recent_device_events"]:
        st.table(state["recent_device_events"])
    else:
        st.caption("기록된 기기 변경이 없습니다.")
    st.subheader("센서 시연 · T06")
    if state["recent_har_events"]:
        preset_labels = {item["preset_id"]: item["label"] for item in state["har_presets"]}
        st.table([
            {
                "시각(UTC)": event["timestamp"],
                "시나리오": preset_labels[event["preset_id"]],
                "T06": T06_DECISION[event["decision"]],
                "사유": event["reason"],
            }
            for event in state["recent_har_events"]
        ])
    else:
        st.caption("기록된 센서 시연이 없습니다.")


def main():
    st.set_page_config(page_title="로컬 AI 스마트홈", layout="wide")
    st.title("로컬 AI 스마트홈")
    user_id = st.sidebar.radio("시연 사용자", ("A", "B"))
    st.sidebar.button("서버 상태 새로고침")
    st.sidebar.caption(f"로컬 API · {API_BASE_URL}")
    health = {}
    try:
        health_response = api_request("/health")
        health = health_response["services"]
        epoch = health_response.get("session_epoch")
        if epoch is not None and st.session_state.get("backend_epoch") != epoch:
            # 새 백엔드/DB 초기화 뒤에는 이전 요청 ID를 전송하지 않는다.
            st.session_state.clear()
            st.session_state["backend_epoch"] = epoch
            st.rerun()
        startup = health_response.get("startup", {})
        if startup:
            st.sidebar.caption("서버 시작 준비 · " + SERVICE_STATUS.get(startup["status"], startup["status"]))
            for step in startup.get("steps", []):
                label = SERVICE_STATUS.get(step["status"], step["status"])
                elapsed = step.get("elapsed_seconds")
                st.sidebar.caption(f"{step['name']} · {label}"
                                   + (f" · {elapsed}초" if elapsed is not None else "")
                                   + (f" · {step['reason']}" if step.get("reason") else ""))
        with st.sidebar.expander("서비스 상태", expanded=True):
            for name, service in health.items():
                label = SERVICE_STATUS[service["status"]]
                reason = service.get("reason")
                st.text(f"{name} · {label}" + (f" ({reason})" if reason else ""))
    except ApiError as error:
        st.sidebar.error(str(error))

    profiles = st.session_state.setdefault("profiles", {})
    profile = profiles.setdefault(user_id, {"messages": [], "pending": None, "retry_reason": None})
    state = None
    try:
        state = api_request("/state", user_id=user_id)
    except ApiError as error:
        st.error("현재 집 상태 확인 불가 · " + str(error))
    conversation, house, activity, logs = st.tabs(("대화", "가상 집", "시연", "실행 로그"))
    with conversation:
        render_chat(user_id, profile, state)
    with house:
        if state is not None:
            render_house(state)
        else:
            st.warning("현재 집 상태 확인 불가 · 집 상태를 다시 조회하세요.")
    with activity:
        sensors, personalization = st.tabs(("센서·HAR", "사용 기록·개인화"))
        with sensors:
            if state is not None:
                render_activity(user_id, state, health.get("har", {}))
            else:
                st.warning("센서 시나리오를 조회할 수 없습니다.")
        with personalization:
            render_personalization(user_id, profile, state)
    with logs:
        if state is not None:
            render_logs(user_id, state)


if __name__ == "__main__":
    main()
