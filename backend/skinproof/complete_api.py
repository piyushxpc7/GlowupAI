from __future__ import annotations

import base64
import binascii
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .auth import AuthError, verify_id_token
from .complete_service import CompleteSkinProofService
from .config import Settings
from .complete_db import build_full_database
from .photos import build_photo_store


class UserCreate(BaseModel):
    skin_type: str | None = None


class ConsentCreate(BaseModel):
    facial_data: bool
    policy_version: str | None = None


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    barcode: str | None = None
    category: str = "other"
    ingredients: list[str] | str | None = None
    stabilization_days: int = Field(default=14, ge=0, le=180)


class RoutineEventCreate(BaseModel):
    user_id: str
    product_id: str
    action: str
    timestamp: str | None = None
    slot: str = "unspecified"
    dose: str | None = None
    frequency: str | None = None
    notes: str | None = None
    experiment_id: str | None = None


class CaptureCreate(BaseModel):
    user_id: str
    image_base64: str
    quality: dict | None = None
    captured_at: str | None = None
    device_meta: dict | None = None
    is_baseline: bool = False
    vertical: str = "skin"
    experiment_id: str | None = None


class ExperimentCreate(BaseModel):
    user_id: str
    name: str = Field(min_length=1, max_length=160)
    hypothesis: str | None = None
    product_id: str
    primary_metric: str = "redness_score"
    target_days: int = Field(default=14, ge=1, le=180)


class ExperimentStatus(BaseModel):
    user_id: str
    status: str


class QnaCreate(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


class UpgradeCreate(BaseModel):
    source: str = "local_checkout"


class EngagementCreate(BaseModel):
    event_type: str
    reference_id: str | None = None
    metadata: dict | None = None


class OfferCreate(BaseModel):
    product_id: str
    merchant: str
    url: str
    price_cents: int | None = Field(default=None, ge=0)
    currency: str = "USD"


class LabelCreate(BaseModel):
    photo_id: str
    label_type: str
    value: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    notes: str | None = None


class ReprocessCreate(BaseModel):
    model_version: str = Field(min_length=1, max_length=80)


class TriageCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ExperienceProfileUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)
    skin_type: str | None = Field(default=None, max_length=40)
    focus_vertical: str | None = None
    goals: list[str] | None = None
    experience_level: str | None = Field(default=None, max_length=80)
    onboarding_complete: bool | None = None


class ShelfScanCreate(BaseModel):
    image_base64: str


class ShelfScanConfirm(BaseModel):
    selections: list[dict]


class ContextEventCreate(BaseModel):
    event_type: str
    value: str | None = None
    occurred_at: str | None = None
    notes: str | None = None


class CheckInCreate(BaseModel):
    routine_state: str = Field(default="steady", pattern="^(steady|changed|missed|not_sure)$")
    skin_feel: str = Field(default="not_sure", pattern="^(better|same|worse|not_sure)$")
    note: str | None = Field(default=None, max_length=400)
    occurred_at: str | None = None


class MeasurementFeedbackCreate(BaseModel):
    capture_id: str
    agreement: str = Field(pattern="^(fair|uncertain|off)$")
    note: str | None = Field(default=None, max_length=400)


class PurchaseGuidanceCreate(BaseModel):
    name: str | None = Field(default=None, max_length=160)
    barcode: str | None = Field(default=None, max_length=80)
    category: str = Field(default="other", max_length=40)
    ingredients: list[str] | str | None = None
    price_cents: int | None = Field(default=None, ge=0)
    currency: str = Field(default="INR", min_length=3, max_length=3)

def create_complete_app(service: CompleteSkinProofService | None = None) -> FastAPI:
    settings = Settings.from_env()
    settings.prepare()
    active = service or CompleteSkinProofService(build_full_database(settings), settings, build_photo_store(settings.photo_dir))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        active.jobs.shutdown()

    app = FastAPI(
        title="SkinProof",
        version="3.0.0",
        description="A complete personal appearance measurement system. Cosmetic tracking, never diagnosis.",
        lifespan=lifespan,
    )
    app.state.skinproof = active
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def run(callable_, *args, **kwargs):
        try:
            return callable_(*args, **kwargs)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            detail = str(exc)
            try:
                detail = json.loads(detail)
            except json.JSONDecodeError:
                pass
            raise HTTPException(status_code=400, detail=detail) from exc

    # -- auth boundary --------------------------------------------------------
    #
    # When `SKINPROOF_AUTH_REQUIRED` is off (the default), `_require_owner` is
    # a no-op and every route behaves exactly as it did before this module
    # existed. That default is deliberate: the existing test suite and the
    # unauthenticated static/web client carry no bearer token and must keep
    # working unchanged. Flip the flag on only once a deployment has a real
    # Firebase project configured and every client sends a token.

    def _bearer_identity(authorization: str | None):
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        try:
            return verify_id_token(token, active.settings.firebase_project_id)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def _require_owner(user_id: str, authorization: str | None) -> None:
        """No-op unless SKINPROOF_AUTH_REQUIRED is set; then 401/403 on mismatch."""

        if not active.settings.auth_required:
            return
        identity = _bearer_identity(authorization)
        row = active.db.fetchone("SELECT id FROM users WHERE firebase_uid = ?", (identity.uid,))
        if not row or row["id"] != user_id:
            raise HTTPException(status_code=403, detail="the authenticated account does not own this user_id")

    def _require_admin(authorization: str | None) -> None:
        if not active.settings.admin_token:
            raise HTTPException(status_code=403, detail="admin endpoints are disabled: SKINPROOF_ADMIN_TOKEN is not configured")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=403, detail="missing admin bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if not secrets.compare_digest(token, active.settings.admin_token):
            raise HTTPException(status_code=403, detail="invalid admin token")

    @app.get("/api/health")
    def health():
        try:
            database_ready = active.db.healthcheck()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
        return {"status": "ok", "database": getattr(active.db, "backend", "sqlite"), "database_ready": database_ready, "version": "3.0.0", "scope": "cosmetic_tracking", "features": ["experiments", "qna", "discover", "commerce", "reprocessing", "shelf_scan", "product_prediction", "root_cause_search", "budget_optimizer", "derm_export"]}

    @app.post("/api/users")
    @limiter.limit("20/minute")
    def create_user(request: Request, payload: UserCreate):
        return active.create_user(payload.skin_type)

    @app.post("/api/auth/session")
    @limiter.limit("30/minute")
    def auth_session(request: Request, authorization: str | None = Header(default=None)):
        identity = _bearer_identity(authorization)
        return run(active.session_for_identity, identity.uid, identity.email, identity.email_verified, identity.name)

    @app.get("/api/users/{user_id}/profile")
    def profile(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.profile, user_id)

    @app.patch("/api/users/{user_id}/profile", tags=["profile"])
    def update_experience_profile(user_id: str, payload: ExperienceProfileUpdate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.update_profile, user_id, **payload.model_dump())

    @app.post("/api/users/{user_id}/consent")
    def consent(user_id: str, payload: ConsentCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.grant_consent, user_id, payload.facial_data, payload.policy_version)

    @app.get("/api/users/{user_id}/subscription")
    def subscription(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.entitlement, user_id)

    @app.post("/api/users/{user_id}/subscription/upgrade")
    def upgrade(user_id: str, payload: UpgradeCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.upgrade, user_id, payload.source)

    @app.post("/api/users/{user_id}/subscription/cancel")
    def cancel_subscription(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.downgrade, user_id)

    @app.post("/api/products")
    def create_product(payload: ProductCreate):
        return run(active.create_product, payload.name, payload.barcode, payload.category, payload.ingredients, payload.stabilization_days)

    @app.get("/api/products/search")
    def search_products(q: str = ""):
        return active.search_products(q)

    @app.get("/api/products/lookup")
    def lookup_product(barcode: str):
        return run(active.lookup_product, barcode)
    @app.get("/api/products/{product_id}")
    def product_detail(product_id: str, user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.product_detail, user_id, product_id)

    @app.get("/api/products/{product_id}/ingredient-explainer")
    def ingredient_explainer(product_id: str, user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.ingredient_explainer, user_id, product_id)

    @app.get("/api/products/{product_id}/predict")
    def predict_product(product_id: str, user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.predict_product, user_id, product_id)

    @app.post("/api/users/{user_id}/purchase-guidance")
    def purchase_guidance(user_id: str, payload: PurchaseGuidanceCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.purchase_guidance, user_id, **payload.model_dump())
    @app.post("/api/routine-events")
    def routine_event(payload: RoutineEventCreate, authorization: str | None = Header(default=None)):
        _require_owner(payload.user_id, authorization)
        return run(active.add_routine_event, **payload.model_dump())

    @app.get("/api/users/{user_id}/confound-check")
    def confound_check(user_id: str, exclude_product_id: str | None = None, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.confound_check, user_id, exclude_product_id)

    @app.post("/api/captures")
    @limiter.limit("30/minute")
    def capture(payload: CaptureCreate, request: Request, authorization: str | None = Header(default=None)):
        _require_owner(payload.user_id, authorization)
        try:
            image = base64.b64decode(payload.image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="image_base64 must be valid base64") from exc
        return run(active.create_capture, payload.user_id, image, payload.quality, payload.captured_at, payload.device_meta, payload.is_baseline, payload.vertical, payload.experiment_id)

    @app.get("/api/users/{user_id}/capture-guide")
    def capture_guide(user_id: str, vertical: str = "skin", authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.capture_guide, user_id, vertical)

    @app.get("/api/users/{user_id}/dashboard")
    def dashboard(user_id: str, vertical: str = "skin", authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.dashboard, user_id, vertical)

    @app.get("/api/users/{user_id}/history")
    def history(user_id: str, vertical: str = "skin", authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.history, user_id, vertical)

    @app.get("/api/users/{user_id}/engagement")
    def engagement(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.engagement, user_id)

    @app.post("/api/users/{user_id}/engagement")
    def engagement_event(user_id: str, payload: EngagementCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.record_engagement, user_id, payload.event_type, payload.reference_id, payload.metadata)

    @app.get("/api/users/{user_id}/check-ins")
    def check_ins(user_id: str, limit: int = 30, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.check_ins, user_id, limit)

    @app.post("/api/users/{user_id}/check-ins")
    def create_check_in(user_id: str, payload: CheckInCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.create_check_in, user_id, **payload.model_dump())

    @app.get("/api/users/{user_id}/weekly-recap")
    def weekly_recap(user_id: str, vertical: str = "skin", as_of: str | None = None, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.weekly_recap, user_id, vertical, as_of)

    @app.post("/api/users/{user_id}/measurement-feedback")
    def measurement_feedback(user_id: str, payload: MeasurementFeedbackCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.add_measurement_feedback, user_id, payload.capture_id, payload.agreement, payload.note)
    @app.get("/api/users/{user_id}/analytics")
    def analytics(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.analytics, user_id)

    @app.post("/api/experiments")
    def experiment(payload: ExperimentCreate, authorization: str | None = Header(default=None)):
        _require_owner(payload.user_id, authorization)
        return run(active.create_experiment, payload.user_id, payload.name, payload.hypothesis, payload.product_id, payload.primary_metric, payload.target_days)

    @app.get("/api/users/{user_id}/experiments")
    def experiments(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.experiments, user_id)

    @app.get("/api/users/{user_id}/experiments/{experiment_id}")
    def experiment_detail(user_id: str, experiment_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.experiment, experiment_id, user_id)

    @app.post("/api/users/{user_id}/experiments/{experiment_id}/status")
    def experiment_status(user_id: str, experiment_id: str, payload: ExperimentStatus, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        if payload.user_id != user_id:
            raise HTTPException(status_code=400, detail="user_id mismatch")
        return run(active.set_experiment_status, user_id, experiment_id, payload.status)

    @app.post("/api/users/{user_id}/qna")
    @limiter.limit("20/minute")
    def qna(user_id: str, payload: QnaCreate, request: Request, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.ask, user_id, payload.question, payload.thread_id)

    @app.get("/api/users/{user_id}/qna")
    def qna_history(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.qna_history, user_id)

    @app.get("/api/users/{user_id}/discover")
    def discover(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.discover, user_id)

    @app.get("/api/users/{user_id}/commerce/offers")
    def offers(user_id: str, product_id: str | None = None, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.offers, user_id, product_id)

    @app.post("/api/users/{user_id}/commerce/offers/{offer_id}/click")
    def click_offer(user_id: str, offer_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.click_offer, user_id, offer_id)

    @app.post("/api/admin/offers")
    def add_offer(payload: OfferCreate, authorization: str | None = Header(default=None)):
        _require_admin(authorization)
        return run(active.add_offer, payload.product_id, payload.merchant, payload.url, payload.price_cents, payload.currency)

    @app.get("/api/users/{user_id}/labels")
    def labels(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.labels, user_id)

    @app.post("/api/users/{user_id}/labels")
    def add_label(user_id: str, payload: LabelCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.add_label, user_id, payload.photo_id, payload.label_type, payload.value, payload.confidence, payload.notes)

    @app.post("/api/users/{user_id}/reprocess")
    def reprocess(user_id: str, payload: ReprocessCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.reprocess, user_id, payload.model_version)

    @app.get("/api/users/{user_id}/reprocess/{job_id}")
    def reprocess_status(user_id: str, job_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.reprocess_status, user_id, job_id)

    @app.post("/api/users/{user_id}/shelf-scan")
    @limiter.limit("20/minute")
    def shelf_scan(user_id: str, payload: ShelfScanCreate, request: Request, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        try:
            image = base64.b64decode(payload.image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="image_base64 must be valid base64") from exc
        return run(active.scan_shelf, user_id, image)

    @app.get("/api/users/{user_id}/shelf-scan/{job_id}")
    def shelf_scan_status(user_id: str, job_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.shelf_scan_status, user_id, job_id)

    @app.post("/api/users/{user_id}/shelf-scan/{job_id}/confirm")
    def shelf_scan_confirm(user_id: str, job_id: str, payload: ShelfScanConfirm, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.confirm_shelf_scan, user_id, job_id, payload.selections)

    @app.get("/api/users/{user_id}/context-events")
    def context_events(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.context_events, user_id)

    @app.post("/api/users/{user_id}/context-events")
    def add_context_event(user_id: str, payload: ContextEventCreate, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.add_context_event, user_id, payload.event_type, payload.value, payload.occurred_at, payload.notes)

    @app.get("/api/users/{user_id}/root-cause")
    def root_cause(user_id: str, metric: str = "texture_score", authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.root_cause_search, user_id, metric)

    @app.get("/api/users/{user_id}/budget-optimizer")
    def budget_optimizer(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.budget_optimizer, user_id)

    @app.get("/api/users/{user_id}/derm-export")
    def derm_export(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.dermatologist_report, user_id)

    @app.get("/api/users/{user_id}/export")
    def export_user(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        return run(active.export_user, user_id)

    @app.delete("/api/users/{user_id}", status_code=204)
    def delete_user(user_id: str, authorization: str | None = Header(default=None)):
        _require_owner(user_id, authorization)
        run(active.delete_user, user_id)

    @app.post("/api/triage")
    def triage_question(payload: TriageCreate):
        return active.triage_question(payload.text)

    @app.get("/api/admin/audit")
    def audit(limit: int = 100, authorization: str | None = Header(default=None)):
        _require_admin(authorization)
        return active.admin_audit(limit)

    @app.get("/api/admin/measurement-feedback")
    def measurement_feedback_summary(authorization: str | None = Header(default=None)):
        _require_admin(authorization)
        return active.measurement_feedback_summary()
    static_dir = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")
    static_file = static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_file)

    return app


app = create_complete_app()
