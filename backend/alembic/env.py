from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.db.session import Base
from app.models import *  # noqa: ensure models are registered
from app.core.config import settings

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL_SYNC)

if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(url=settings.DATABASE_URL_SYNC, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
