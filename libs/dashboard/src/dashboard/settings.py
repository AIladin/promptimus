from pydantic_settings import BaseSettings, SettingsConfigDict


class DashboardSettings(BaseSettings):
    base_url: str
    api_token: str

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="promptimus_dashboard_"
    )


settings = DashboardSettings()  # type: ignore
