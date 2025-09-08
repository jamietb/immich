import { BadRequestException, Injectable, ForbiddenException } from '@nestjs/common';
import _ from 'lodash';
import { sql, RawBuilder } from 'kysely';
import { DateTime, Duration } from 'luxon';
import { JOBS_ASSET_PAGINATION_SIZE } from 'src/constants';
import { OnJob } from 'src/decorators';
import { AssetResponseDto, MapAsset, SanitizedAssetResponseDto, mapAsset } from 'src/dtos/asset-response.dto';
import {
  AssetBulkDeleteDto,
  AssetBulkUpdateDto,
  AssetJobName,
  AssetJobsDto,
  AssetStatsDto,
  UpdateAssetDto,
  mapStats,
} from 'src/dtos/asset.dto';
import { AuthDto } from 'src/dtos/auth.dto';
import { AssetStatus, AssetVisibility, JobName, JobStatus, Permission, QueueName } from 'src/enum';
import { BaseService } from 'src/services/base.service';
import { ISidecarWriteJob, JobItem, JobOf } from 'src/types';
import { requireElevatedPermission } from 'src/utils/access';
import { getAssetFiles, getMyPartnerIds, onAfterUnlink, onBeforeLink, onBeforeUnlink } from 'src/utils/asset.util';

@Injectable()
export class AssetService extends BaseService {
/**
 * Visual similarity using CLIP embeddings (smart_search.embedding).
 * Returns nearest neighbors ordered by cosine distance (lowest first = most similar).
 */
async findSimilarAssets(params: {
  userId: string;
  assetId: string;
  types?: ('IMAGE' | 'VIDEO')[];
  limit?: number;
  offset?: number;
  stacksOnly?: boolean;         // only primary per stack (default true)
  minDate?: string | null;      // ISO 8601
  maxDate?: string | null;      // ISO 8601
  albumId?: string | null;      // restrict within an album
  cameraMake?: string | null;   // requires asset_exif join
  cameraModel?: string | null;  // requires asset_exif join
}) {
  const {
    userId,
    assetId,
    types,
    limit = 100,
    offset = 0,
    stacksOnly = true,
    minDate,
    maxDate,
    albumId,
    cameraMake,
    cameraModel,
  } = params;

  // 1) Ownership check
  const owns = await this.databaseRepository.exec(sql`
    SELECT 1 FROM asset WHERE id = ${assetId} AND "ownerId" = ${userId} LIMIT 1
  `);
  if (!owns.rows.length) {
    throw new ForbiddenException('Asset not found or not accessible');
  }

  // 2) Base has embedding?
  const existsRes = await this.databaseRepository.exec<{ ok: boolean }>(sql`
    SELECT EXISTS (
      SELECT 1 FROM smart_search WHERE "assetId" = ${assetId} AND embedding IS NOT NULL
    ) AS ok
  `);
  const ok = Boolean((existsRes.rows as Array<{ ok: boolean }>)[0]?.ok);
  if (!ok) return [];

  // 3) Optional filters
  let typeFilter: RawBuilder<unknown> = sql``;
  if (types && types.length) {
    typeFilter = sql`AND a.type IN (${sql.join(types as any[], sql`,`)})`;
  }

  const dateFilter =
    minDate && maxDate
      ? sql`AND a."localDateTime" BETWEEN ${minDate} AND ${maxDate}`
      : minDate
      ? sql`AND a."localDateTime" >= ${minDate}`
      : maxDate
      ? sql`AND a."localDateTime" <= ${maxDate}`
      : sql``;

  const albumJoin = albumId ? sql`
    JOIN album_asset aa ON aa."assetsId" = a.id AND aa."albumId" = ${albumId}
  ` : sql``;

  // stack primary only (if enabled)
  const stackJoin = stacksOnly ? sql`
    LEFT JOIN stack st ON st.id = a."stackId"
  ` : sql``;
  const stackFilter = stacksOnly ? sql`
    AND (a."stackId" IS NULL OR a.id = st."primaryAssetId")
  ` : sql``;

  // camera EXIF filter (table name is `asset_exif` with FK "assetId")
const exifJoin =
  cameraMake || cameraModel
    ? sql`LEFT JOIN asset_exif ax ON ax."assetId" = a.id`
    : sql``;

let exifFilter: RawBuilder<unknown> = sql``;
if (cameraMake) {
  exifFilter = sql`${exifFilter} AND ax.make = ${cameraMake}`;
}
if (cameraModel) {
  exifFilter = sql`${exifFilter} AND ax.model = ${cameraModel}`;
}

  // 4) KNN query
  const result = await this.databaseRepository.exec(sql`
    WITH base AS (
      SELECT embedding FROM smart_search
      WHERE "assetId" = ${assetId} AND embedding IS NOT NULL
    )
    SELECT
      a.id,
      a.type,
      a."deviceAssetId",
      a."ownerId",
      (s.embedding <-> base.embedding) AS distance
    FROM base
    JOIN smart_search s ON s.embedding IS NOT NULL
    JOIN asset a        ON a.id = s."assetId"
    ${albumJoin}
    ${exifJoin}
    ${stackJoin}
    WHERE s."assetId" <> ${assetId}
      AND a."ownerId" = ${userId}
      ${typeFilter}
      ${dateFilter}
      ${exifFilter}
      ${stackFilter}
    ORDER BY s.embedding <-> base.embedding
    LIMIT ${limit} OFFSET ${offset}
  `);

  return result.rows as Array<{
    id: string;
    type: 'IMAGE' | 'VIDEO';
    deviceAssetId: string | null;
    ownerId: string;
    distance: number;
  }>;
}

  async getStatistics(auth: AuthDto, dto: AssetStatsDto) {
    if (dto.visibility === AssetVisibility.Locked) {
      requireElevatedPermission(auth);
    }

    const stats = await this.assetRepository.getStatistics(auth.user.id, dto);
    return mapStats(stats);
  }

  async getRandom(auth: AuthDto, count: number): Promise<AssetResponseDto[]> {
    const partnerIds = await getMyPartnerIds({
      userId: auth.user.id,
      repository: this.partnerRepository,
      timelineEnabled: true,
    });
    const assets = await this.assetRepository.getRandom([auth.user.id, ...partnerIds], count);
    return assets.map((a) => mapAsset(a, { auth }));
  }

  async getUserAssetsByDeviceId(auth: AuthDto, deviceId: string) {
    return this.assetRepository.getAllByDeviceId(auth.user.id, deviceId);
  }

  async get(auth: AuthDto, id: string): Promise<AssetResponseDto | SanitizedAssetResponseDto> {
    await this.requireAccess({ auth, permission: Permission.AssetRead, ids: [id] });

    const asset = await this.assetRepository.getById(id, {
      exifInfo: true,
      owner: true,
      faces: { person: true },
      stack: { assets: true },
      tags: true,
    });

    if (!asset) {
      throw new BadRequestException('Asset not found');
    }

    if (auth.sharedLink && !auth.sharedLink.showExif) {
      return mapAsset(asset, { stripMetadata: true, withStack: true, auth });
    }

    const data = mapAsset(asset, { withStack: true, auth });

    if (auth.sharedLink) {
      delete data.owner;
    }

    if (data.ownerId !== auth.user.id || auth.sharedLink) {
      data.people = [];
    }

    return data;
  }

  async update(auth: AuthDto, id: string, dto: UpdateAssetDto): Promise<AssetResponseDto> {
    await this.requireAccess({ auth, permission: Permission.AssetUpdate, ids: [id] });

    const { description, dateTimeOriginal, latitude, longitude, rating, ...rest } = dto;
    const repos = { asset: this.assetRepository, event: this.eventRepository };

    let previousMotion: MapAsset | null = null;
    if (rest.livePhotoVideoId) {
      await onBeforeLink(repos, { userId: auth.user.id, livePhotoVideoId: rest.livePhotoVideoId });
    } else if (rest.livePhotoVideoId === null) {
      const asset = await this.findOrFail(id);
      if (asset.livePhotoVideoId) {
        previousMotion = await onBeforeUnlink(repos, { livePhotoVideoId: asset.livePhotoVideoId });
      }
    }

    await this.updateMetadata({ id, description, dateTimeOriginal, latitude, longitude, rating });

    const asset = await this.assetRepository.update({ id, ...rest });

    if (previousMotion && asset) {
      await onAfterUnlink(repos, {
        userId: auth.user.id,
        livePhotoVideoId: previousMotion.id,
        visibility: asset.visibility,
      });
    }

    if (!asset) {
      throw new BadRequestException('Asset not found');
    }

    return mapAsset(asset, { auth });
  }

  async updateAll(auth: AuthDto, dto: AssetBulkUpdateDto): Promise<void> {
    const { ids, description, dateTimeOriginal, dateTimeRelative, timeZone, latitude, longitude, ...options } = dto;
    await this.requireAccess({ auth, permission: Permission.AssetUpdate, ids });

    const staticValuesChanged =
      description !== undefined || dateTimeOriginal !== undefined || latitude !== undefined || longitude !== undefined;

    if (staticValuesChanged) {
      await this.assetRepository.updateAllExif(ids, { description, dateTimeOriginal, latitude, longitude });
    }

    const assets =
      (dateTimeRelative !== undefined && dateTimeRelative !== 0) || timeZone !== undefined
        ? await this.assetRepository.updateDateTimeOriginal(ids, dateTimeRelative, timeZone)
        : null;

    const dateTimesWithTimezone =
      assets?.map((asset) => {
        const isoString = asset.dateTimeOriginal?.toISOString();
        let dateTime = isoString ? DateTime.fromISO(isoString) : null;

        if (dateTime && asset.timeZone) {
          dateTime = dateTime.setZone(asset.timeZone);
        }

        return {
          assetId: asset.assetId,
          dateTimeOriginal: dateTime?.toISO() ?? null,
        };
      }) ?? null;

    if (staticValuesChanged || dateTimesWithTimezone) {
      const entries: JobItem[] = (dateTimesWithTimezone ?? ids).map((entry: any) => ({
        name: JobName.SidecarWrite,
        data: {
          id: entry.assetId ?? entry,
          description,
          dateTimeOriginal: entry.dateTimeOriginal ?? dateTimeOriginal,
          latitude,
          longitude,
        },
      }));
      await this.jobRepository.queueAll(entries);
    }

    if (
      options.visibility !== undefined ||
      options.isFavorite !== undefined ||
      options.duplicateId !== undefined ||
      options.rating !== undefined
    ) {
      await this.assetRepository.updateAll(ids, options);

      if (options.visibility === AssetVisibility.Locked) {
        await this.albumRepository.removeAssetsFromAll(ids);
      }
    }
  }

  @OnJob({ name: JobName.AssetDeleteCheck, queue: QueueName.BackgroundTask })
  async handleAssetDeletionCheck(): Promise<JobStatus> {
    const config = await this.getConfig({ withCache: false });
    const trashedDays = config.trash.enabled ? config.trash.days : 0;
    const trashedBefore = DateTime.now()
      .minus(Duration.fromObject({ days: trashedDays }))
      .toJSDate();

    let chunk: Array<{ id: string; isOffline: boolean }> = [];
    const queueChunk = async () => {
      if (chunk.length > 0) {
        await this.jobRepository.queueAll(
          chunk.map(({ id, isOffline }) => ({
            name: JobName.AssetDelete,
            data: { id, deleteOnDisk: !isOffline },
          })),
        );
        chunk = [];
      }
    };

    const assets = this.assetJobRepository.streamForDeletedJob(trashedBefore);
    for await (const asset of assets) {
      chunk.push(asset);
      if (chunk.length >= JOBS_ASSET_PAGINATION_SIZE) {
        await queueChunk();
      }
    }

    await queueChunk();

    return JobStatus.Success;
  }

  @OnJob({ name: JobName.AssetDelete, queue: QueueName.BackgroundTask })
  async handleAssetDeletion(job: JobOf<JobName.AssetDelete>): Promise<JobStatus> {
    const { id, deleteOnDisk } = job;

    const asset = await this.assetJobRepository.getForAssetDeletion(id);

    if (!asset) {
      return JobStatus.Failed;
    }

    // Replace the parent of the stack children with a new asset
    if (asset.stack?.primaryAssetId === id) {
      const stackAssetIds = asset.stack?.assets.map((a) => a.id) ?? [];
      if (stackAssetIds.length > 2) {
        const newPrimaryAssetId = stackAssetIds.find((a) => a !== id)!;
        await this.stackRepository.update(asset.stack.id, {
          id: asset.stack.id,
          primaryAssetId: newPrimaryAssetId,
        });
      } else {
        await this.stackRepository.delete(asset.stack.id);
      }
    }

    await this.assetRepository.remove(asset);
    if (!asset.libraryId) {
      await this.userRepository.updateUsage(asset.ownerId, -(asset.exifInfo?.fileSizeInByte || 0));
    }

    await this.eventRepository.emit('AssetDelete', { assetId: id, userId: asset.ownerId });

    // delete the motion if it is not used by another asset
    if (asset.livePhotoVideoId) {
      const count = await this.assetRepository.getLivePhotoCount(asset.livePhotoVideoId);
      if (count === 0) {
        await this.jobRepository.queue({
          name: JobName.AssetDelete,
          data: { id: asset.livePhotoVideoId, deleteOnDisk },
        });
      }
    }

    const { fullsizeFile, previewFile, thumbnailFile } = getAssetFiles(asset.files ?? []);
    const files = [thumbnailFile?.path, previewFile?.path, fullsizeFile?.path, asset.encodedVideoPath];

    if (deleteOnDisk) {
      files.push(asset.sidecarPath, asset.originalPath);
    }

    await this.jobRepository.queue({ name: JobName.FileDelete, data: { files } });

    return JobStatus.Success;
  }

  async deleteAll(auth: AuthDto, dto: AssetBulkDeleteDto): Promise<void> {
    const { ids, force } = dto;

    await this.requireAccess({ auth, permission: Permission.AssetDelete, ids });
    await this.assetRepository.updateAll(ids, {
      deletedAt: new Date(),
      status: force ? AssetStatus.Deleted : AssetStatus.Trashed,
    });
    await this.eventRepository.emit(force ? 'AssetDeleteAll' : 'AssetTrashAll', {
      assetIds: ids,
      userId: auth.user.id,
    });
  }

  async run(auth: AuthDto, dto: AssetJobsDto) {
    await this.requireAccess({ auth, permission: Permission.AssetUpdate, ids: dto.assetIds });

    const jobs: JobItem[] = [];

    for (const id of dto.assetIds) {
      switch (dto.name) {
        case AssetJobName.REFRESH_FACES: {
          jobs.push({ name: JobName.AssetDetectFaces, data: { id } });
          break;
        }

        case AssetJobName.REFRESH_METADATA: {
          jobs.push({ name: JobName.AssetExtractMetadata, data: { id } });
          break;
        }

        case AssetJobName.REGENERATE_THUMBNAIL: {
          jobs.push({ name: JobName.AssetGenerateThumbnails, data: { id } });
          break;
        }

        case AssetJobName.TRANSCODE_VIDEO: {
          jobs.push({ name: JobName.AssetEncodeVideo, data: { id } });
          break;
        }
      }
    }

    await this.jobRepository.queueAll(jobs);
  }

  private async findOrFail(id: string) {
    const asset = await this.assetRepository.getById(id);
    if (!asset) {
      throw new BadRequestException('Asset not found');
    }
    return asset;
  }

  private async updateMetadata(dto: ISidecarWriteJob) {
    const { id, description, dateTimeOriginal, latitude, longitude, rating } = dto;
    const writes = _.omitBy({ description, dateTimeOriginal, latitude, longitude, rating }, _.isUndefined);
    if (Object.keys(writes).length > 0) {
      await this.assetRepository.upsertExif({ assetId: id, ...writes });
      await this.jobRepository.queue({ name: JobName.SidecarWrite, data: { id, ...writes } });
    }
  }
}
