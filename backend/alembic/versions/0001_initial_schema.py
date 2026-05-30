"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-30
"""
from alembic import op
import sqlalchemy as sa

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ────────────────────────────────────────────────────────────────
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_users_id', 'users', ['id'])
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    # ── papers ───────────────────────────────────────────────────────────────
    op.create_table(
        'papers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('authors', sa.Text(), nullable=True),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('year', sa.Integer(), nullable=True),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('faiss_index_id', sa.String(100), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='processing'),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('paper_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_papers_id', 'papers', ['id'])
    op.create_index('ix_papers_user_id', 'papers', ['user_id'])

    # ── paper_chunks ─────────────────────────────────────────────────────────
    op.create_table(
        'paper_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('paper_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('faiss_vector_id', sa.Integer(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_paper_chunks_id', 'paper_chunks', ['id'])
    op.create_index('ix_paper_chunks_paper_id', 'paper_chunks', ['paper_id'])

    # ── query_history ────────────────────────────────────────────────────────
    op.create_table(
        'query_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=True),
        sa.Column('query_type', sa.String(50), nullable=False, server_default='qa'),
        sa.Column('paper_ids', sa.JSON(), nullable=True),
        sa.Column('citations', sa.JSON(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_query_history_id', 'query_history', ['id'])
    op.create_index('ix_query_history_user_id', 'query_history', ['user_id'])


def downgrade() -> None:
    op.drop_table('query_history')
    op.drop_table('paper_chunks')
    op.drop_table('papers')
    op.drop_table('users')
